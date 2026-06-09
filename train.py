import os
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "./lora-qwen-traffic-all-tasks"
DATA_PATH = "train/all_tasks_merged.json" 

def format_vlm_prompt(examples):
    texts = []
    
    zipped_data = zip(
        examples['video_id'], 
        examples['question'], 
        examples['answer'], 
        examples['task_type']
    )
    
    for video_path, question, answer, task_type in zipped_data:
        sys_prompt = "You are a traffic anomaly expert."
        
        if task_type == "temporal_localization":
            sys_prompt += " Provide exact temporal boundaries in strict JSON format: {'start': X, 'end': Y}."
        elif task_type in ["bcq", "bcq_openended"]:
            sys_prompt += " Answer strictly with 'Yes' or 'No'."
        elif task_type in ["mcq", "mcq_openended"]:
            sys_prompt += " Select the best option. Answer concisely."
        else:
            sys_prompt += " Analyze the video and answer the question in detail."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                # Increased FPS to 2.0 to capture finer temporal details for mIoU metrics
                {"type": "video", "video": f"train/videos/{video_path}.mp4", "fps": 2.0},
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

print("Loading Qwen2-VL Model and Processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Increased LoRA capacity (r=64) to handle 10 different tasks simultaneously
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Loading merged multi-task dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(format_vlm_prompt, batched=True, remove_columns=dataset.column_names)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, 
    learning_rate=2e-5, 
    lr_scheduler_type="cosine", # Added Cosine Scheduler
    warmup_ratio=0.05,          # Added Warmup 
    logging_steps=20,
    max_steps=5000, 
    save_steps=1000,
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

print("Starting Full Multi-Task VLM LoRA Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)