import os
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "./lora-qwen-traffic-all-tasks"
DATA_PATH = "all_tasks_merged.json" 

def format_vlm_prompt(example):
    # Process a single example to return raw messages, allowing the DataCollator to handle video loading
    video_path = example['video_id']
    question = example['question']
    answer = example['answer']
    task_type = example['task_type']

    sys_prompt = "You are an expert AI system specialized in traffic anomaly detection and reasoning. Your task is to analyze the provided traffic video and respond based on the specific constraints."
    
    if task_type == "temporal_localization":
        sys_prompt += " Analyze the video to identify the exact onset and conclusion of the anomaly. Provide your answer strictly as a JSON object with 'start' and 'end' keys representing seconds. Example: {\"start\": 12.5, \"end\": 45.0}. Do NOT include any other text, reasoning, or markdown formatting."
    elif task_type in ["bcq", "bcq_openended"]:
        sys_prompt += " Answer strictly with 'Yes' or 'No' based solely on the visual evidence."
    elif task_type in ["mcq", "mcq_openended"]:
        sys_prompt += " Select the best option. Answer concisely by providing the exact text of the correct option."
    else:
        sys_prompt += " Think step-by-step: first identify the vehicles or pedestrians involved, describe their actions chronologically, and finally deduce the anomaly to answer the question in detail."

    full_video_path = f"/media/RAID5Array/backup_home/tindd4/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/train/videos/{video_path}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "video", "video": full_video_path, "fps": 2.0},
            {"type": "text", "text": question}
        ]},
        {"role": "assistant", "content": answer}
    ]
    
    return {"messages": messages}

class QwenVLDataCollator:
    """Custom collator to ensure video paths are correctly processed into pixel values during training."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        messages_list = [feature["messages"] for feature in features]
        
        # Apply chat template correctly to generate the text prompt for the batch
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages_list
        ]
        
        # Extract images and videos natively via qwen_vl_utils
        image_inputs_list, video_inputs_list = [], []
        for msg in messages_list:
            img, vid = process_vision_info(msg)
            image_inputs_list.append(img)
            video_inputs_list.append(vid)
            
        # Flatten lists for processor
        images = [img for sublist in image_inputs_list if sublist is not None for img in sublist] if any(image_inputs_list) else None
        videos = [vid for sublist in video_inputs_list if sublist is not None for vid in sublist] if any(video_inputs_list) else None

        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt"
        )

        # In standard Causal LM, labels are the same as input_ids. 
        # TRL handles prompt-masking if configured, but a simple shift is standard.
        batch["labels"] = batch["input_ids"].clone()
        return batch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Qwen2-VL Model and Processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Loading and mapping multi-task dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
# Use map with batched=False for easier dict processing, keep column names for now
dataset = dataset.map(format_vlm_prompt, batched=False, remove_columns=dataset.column_names)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, 
    learning_rate=2e-5, 
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,            # Prefer warmup_ratio over absolute steps for flexibility
    logging_steps=20,
    max_steps=5000, 
    save_steps=1000,
    save_total_limit=2,           # Added to prevent massive storage usage
    bf16=True,
    optim="paged_adamw_8bit",
    weight_decay=0.01,            # Added standard weight decay
    remove_unused_columns=False,  # CRITICAL: Do not let HF strip video paths before the collator
    max_seq_length=4096           # Added to prevent unexpected video token truncation
)

# Instantiate the custom collator
data_collator = QwenVLDataCollator(processor)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,   # Use the custom collator to properly fetch videos
)

print("Starting Full Multi-Task VLM LoRA Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR) # Save processor along with the model