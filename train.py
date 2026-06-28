"""
train.py — Improved Qwen3-VL Fine-tuning with Proper Video Processing
=====================================================================

KEY FIXES vs original:
1. [CRITICAL] Custom VideoQADataset + VideoCollator actually processes video frames
   via process_vision_info(). The original SFTTrainer with dataset_text_field="text"
   never loads pixel values — the model never saw real video during training.
2. [CRITICAL] Proper label masking: loss is computed only on the assistant's
   response tokens, not on the entire sequence (including system/user prompt).
3. Temporal localization format fixed to strict double-quoted JSON.
4. fps=1.0 + max_pixels cap keeps memory bounded for long surveillance videos.
5. Replaced SFTTrainer with plain Trainer for full control over collation.
6. Chain-of-thought answers retained (dataset has CoT traces).
7. [CRITICAL FIX] Added mm_token_type_ids extraction and padding for M-RoPE support.
"""

import os
import re
import json
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_ID        = "Qwen/Qwen3-VL-8B-Instruct"   # 7B > 4B for open-ended F1
OUTPUT_DIR      = "./lora-qwen3-traffic-v2"
DATA_PATH       = "all_tasks_merged.json"
VIDEO_BASE_DIR  = "/media/RAID5Array/pdcuong/PhysicalAI-Traffic-Anomaly-Reasoning/videos"
MAX_LENGTH      = 3072         # longer context captures more video frames
VIDEO_FPS       = 1.0          # 1 frame/sec → manageable token budget for long clips
VIDEO_MAX_PIXELS = 360 * 480   # ~170K pixels per frame


# ─── System Prompts ───────────────────────────────────────────────────────────
def get_system_prompt(task_type: str) -> str:
    base = "You are an expert traffic surveillance analyst."
    prompts = {
        "bcq": (
            base + " You will be shown a traffic video and asked a yes/no question. "
            "Respond with exactly one word: 'Yes' or 'No'. Nothing else."
        ),
        "bcq_openended": (
            base + " You will be shown a traffic video. First answer 'Yes' or 'No', "
            "then provide a detailed chain-of-thought explanation of your reasoning, "
            "citing specific visual evidence from the video."
        ),
        "mcq": (
            base + " You will be shown a traffic video and a multiple-choice question. "
            "Output only the letter (A, B, C, or D) of the single best answer. Nothing else."
        ),
        "mcq_openended": (
            base + " You will be shown a traffic video and a multiple-choice question. "
            "First output the letter (A, B, C, or D) of the best answer, then explain "
            "your reasoning step by step with evidence from the video."
        ),
        "temporal_localization": (
            base + " You will be shown a traffic video and asked to localize an event "
            "in time. Respond ONLY with valid JSON using double quotes in this exact "
            'format: {"start": <number>, "end": <number>}  — where start and end are '
            "timestamps in seconds (floats allowed). No other text."
        ),
        "open_qa": (
            base + " You will be shown a traffic video. Answer the question with a "
            "comprehensive, detailed response that references specific visual evidence, "
            "vehicle movements, and temporal context you observe in the video."
        ),
        "causal_linkage": (
            base + " You will be shown a traffic video. Provide a detailed causal "
            "analysis that explains the chain of events: what happened, why it happened, "
            "what factors contributed, and what the consequences were. Reference specific "
            "visual evidence."
        ),
        "scene_description": (
            base + " You will be shown a traffic video. Provide a comprehensive scene "
            "description covering: the road type and environment, all vehicles and their "
            "movements, any anomalies or incidents, and the overall traffic context."
        ),
        "temporal_description": (
            base + " You will be shown a traffic video. Provide a detailed temporal "
            "narrative describing what happens chronologically, including when specific "
            "events occur, how the scene evolves, and the sequence of key moments."
        ),
        "video_summarization": (
            base + " You will be shown a traffic video. Provide a thorough summary "
            "covering: the main events, any anomalies or incidents, vehicle behaviors, "
            "and the overall significance of what occurred."
        ),
    }
    return prompts.get(task_type, base + " Analyze the traffic video and answer in detail.")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class VideoQADataset(Dataset):
    def __init__(
        self,
        data_list: List[Dict],
        processor: Any,
        video_base_dir: str,
    ):
        self.data = data_list
        self.processor = processor
        self.video_base_dir = video_base_dir

    def __len__(self):
        return len(self.data)

    def _build_messages(self, item: Dict) -> List[Dict]:
        video_id  = item["video_id"]
        question  = item["question"]
        answer    = item["answer"]
        task_type = item.get("task_type", "open_qa")

        # Clean temporal localization answers to strict JSON
        if task_type == "temporal_localization":
            answer = _normalize_temporal_answer(answer)

        video_path = os.path.join(self.video_base_dir, video_id)

        messages = [
            {"role": "system", "content": get_system_prompt(task_type)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": VIDEO_FPS,
                        "max_pixels": VIDEO_MAX_PIXELS,
                    },
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
        return messages

    def __getitem__(self, idx: int) -> Dict:
        item     = self.data[idx]
        messages = self._build_messages(item)

        # Apply chat template — this produces the full <|im_start|>…<|im_end|> text
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Also build the prompt-only text to know where the assistant answer starts
        prompt_messages = messages[:-1]  # drop assistant turn
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize full conversation
        encoding = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        # Tokenize prompt to find the length to mask
        prompt_encoding = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        input_ids   = encoding["input_ids"][0]
        prompt_len  = prompt_encoding["input_ids"].shape[1]

        # ── Label masking: only compute loss on the assistant response ──
        labels = input_ids.clone()
        labels[:prompt_len] = -100   # mask system + user + "<|im_start|>assistant\n"
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # -- FIX: Unpack 1D sequence tensors, including mm_token_type_ids --
        result = {}
        for k, v in encoding.items():
            if k in ["input_ids", "attention_mask", "mm_token_type_ids"]:
                result[k] = v[0]  
            else:
                result[k] = v     

        result["labels"] = labels
        return result


def _normalize_temporal_answer(answer: str) -> str:
    """Ensure temporal answers use strict double-quoted JSON."""
    # Already valid JSON?
    try:
        obj = json.loads(answer)
        if "start" in obj and "end" in obj:
            return json.dumps({"start": float(obj["start"]), "end": float(obj["end"])})
    except Exception:
        pass

    # Single-quote Python dict → JSON
    try:
        cleaned = answer.replace("'", '"')
        obj = json.loads(cleaned)
        if "start" in obj and "end" in obj:
            return json.dumps({"start": float(obj["start"]), "end": float(obj["end"])})
    except Exception:
        pass

    # Regex fallback
    s = re.search(r'["\']?start["\']?\s*:\s*(\d+(?:\.\d+)?)', answer)
    e = re.search(r'["\']?end["\']?\s*:\s*(\d+(?:\.\d+)?)',   answer)
    if s and e:
        return json.dumps({"start": float(s.group(1)), "end": float(e.group(1))})

    # Can't parse — return as-is and hope fine-tuning corrects it
    return answer


# ─── Collator ─────────────────────────────────────────────────────────────────
class VideoCollator:
    """Pads a batch of variable-length encoded samples to the same length."""

    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)

        batch = {key: [] for key in features[0]}
        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            batch["input_ids"].append(
                torch.nn.functional.pad(f["input_ids"], (0, pad_len), value=self.pad_id)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(
                    f.get("attention_mask", torch.ones(seq_len, dtype=torch.long)),
                    (0, pad_len), value=0,
                )
            )
            batch["labels"].append(
                torch.nn.functional.pad(f["labels"], (0, pad_len), value=-100)
            )
            
            # -- FIX: Pad mm_token_type_ids if present --
            if "mm_token_type_ids" in f:
                batch.setdefault("mm_token_type_ids", []).append(
                    torch.nn.functional.pad(f["mm_token_type_ids"], (0, pad_len), value=0)
                )

            # Vision tensors — include only from first item (batch_size=1 in practice)
            for k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
                if k in f:
                    batch.setdefault(k, []).append(f[k])

        result = {
            "input_ids":      torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels":         torch.stack(batch["labels"]),
        }
        
        # -- FIX: Stack mm_token_type_ids into the batch result --
        if "mm_token_type_ids" in batch and len(batch["mm_token_type_ids"]) > 0:
            result["mm_token_type_ids"] = torch.stack(batch["mm_token_type_ids"])

        # Vision tensors — concat along batch dim when present
        for k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            if k in batch:
                result[k] = torch.cat(batch[k], dim=0)

        return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Quantisation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {MODEL_ID} …")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        # attn_implementation="flash_attention_2",  # enable if flash-attn installed
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA — keep r=64 (r=128 costs 2x VRAM with diminishing returns for 7B)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    print("Loading dataset …")
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)

    # Filter out missing videos
    valid_data = []
    for item in raw_data:
        video_path = os.path.join(VIDEO_BASE_DIR, item["video_id"])
        if os.path.exists(video_path):
            valid_data.append(item)
        else:
            print(f"  ⚠ Skipping missing training video: {video_path}")

    print(f"Loaded {len(valid_data)} valid training samples (skipped {len(raw_data) - len(valid_data)}).")
    train_dataset = VideoQADataset(valid_data, processor, VIDEO_BASE_DIR)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,   # always 1 — video frames are huge
        gradient_accumulation_steps=16,  # effective batch = 16
        learning_rate=1e-4,              # higher LR works well for LoRA
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        num_train_epochs=3,
        max_steps=-1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        dataloader_num_workers=2,        # 2 is safer for video loading
        remove_unused_columns=False,     # CRITICAL — keeps vision keys
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VideoCollator(pad_token_id=processor.tokenizer.pad_token_id),
    )

    print("Starting training …")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()