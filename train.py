import os
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "./lora-qwen-traffic-reasoning"
# Assuming videos are mapped correctly in your dataset JSON
DATA_PATH = "train/temporal_localization_cleaned.json" 

def format_vlm_prompt(examples):
    """Formats the dataset for Qwen2-VL video ingestion."""
    texts = []
    for video_path, question, answer in zip(examples['video_id'], examples['question'], examples['answer']):
        # We instruct the model precisely for JSON output to optimize temporal localization
        sys_prompt = "You are a traffic anomaly expert. Provide exact temporal boundaries in strict JSON format: {'start': X, 'end': Y}."
        
        # Qwen2-VL specific chat template format
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                # Sample video at 1 FPS to fit into context window memory constraints
                {"type": "video", "video": f"videos/{video_path}.mp4", "fps": 1.0},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": answer}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Model and Processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(format_vlm_prompt, batched=True, remove_columns=dataset.column_names)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, 
    learning_rate=2e-5, # Lower LR for VLMs compared to text-only LLMs
    logging_steps=10,
    max_steps=2000,
    save_steps=400,
    bf16=True,
    optim="paged_adamw_8bit",
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
    tokenizer=processor.tokenizer,
)

print("Starting VLM LoRA Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)