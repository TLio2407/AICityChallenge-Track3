import os
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "./lora-qwen-traffic-all-tasks"
DATA_PATH = "all_tasks_merged.json" 

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
        
        # STRICT System Prompts
        if task_type == "temporal_localization":
            sys_prompt += " Output the final answer as a strict JSON: {\"start\": X.X, \"end\": Y.Y}. No other text."
        elif task_type in ["bcq", "bcq_openended"]:
            sys_prompt += " You must answer strictly with a single word: 'Yes' or 'No'. Do not explain."
        elif task_type in ["mcq", "mcq_openended"]:
            sys_prompt += " Answer strictly with the exact option provided. Do not include reasoning."
        else:
            sys_prompt += " Analyze the video and answer the question in detail."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": f"/media/RAID5Array/backup_home/tindd4/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/train/videos/{video_path}", "fps": 2.0},
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

# Define Completion-Only Collator to mask prompt loss during backpropagation
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=processor.tokenizer)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, 
    learning_rate=2e-5, 
    lr_scheduler_type="cosine",
    warmup_steps=0.05,          
    logging_steps=20,
    max_steps=5000, 
    save_steps=1000,
    bf16=True,
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    neftune_noise_alpha=5.0  # Added for embedding regularization and improved F1
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=processor,
    data_collator=collator, # Apply prompt masking
)

print("Starting Full Multi-Task VLM LoRA Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)