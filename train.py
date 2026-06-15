import os
import torch
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# UPDATED: Target the new Thinking model
MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
OUTPUT_DIR = "./lora-qwen3-traffic-all-tasks"
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
        
        if task_type == "bcq":
            sys_prompt += " Answer strictly with exactly one word: 'Yes' or 'No'."
        elif task_type == "bcq_openended":
            sys_prompt += " Begin your answer with 'Yes' or 'No', then provide a detailed reasoning explanation."
        elif task_type == "mcq":
            sys_prompt += " Output only the single letter (A, B, C, or D) of the correct answer."
        elif task_type == "mcq_openended":
            sys_prompt += " Begin with the letter of the correct answer (A, B, C, or D), then explain your reasoning in detail."
        elif task_type == "temporal_localization":
            sys_prompt += ' Provide the anomaly interval as JSON: {"start": "MM:SS", "end": "MM:SS"}.'
        elif task_type == "causal_linkage":
            sys_prompt += " Identify and explain the causal chain of events that led to the anomaly."
        elif task_type == "scene_description":
            sys_prompt += " Describe the scene in detail, including all visible elements and their spatial relationships."
        elif task_type == "video_summarization":
            sys_prompt += " Provide a comprehensive summary of all events occurring in the video."
        elif task_type == "temporal_description":
            sys_prompt += " Describe the sequence of events with close attention to temporal order and timing."
        else:  # open_qa
            sys_prompt += " Analyze the video carefully and answer the question with detailed reasoning."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {
                    "type": "video", 
                    "video": f"/media/RAID5Array/backup_home/tindd4/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/train/videos/{video_path}",
                    "fps": 1.0,           
                    "max_pixels": 256000  
                },
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": answer}
        ]
        
        # Apply Chat Template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        
    return {"text": texts}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Qwen3-VL Model and Processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
# UPDATED: New Qwen3 class
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2" 
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=128, 
    lora_alpha=256, 
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
    per_device_train_batch_size=4,     
    gradient_accumulation_steps=4,     
    dataloader_num_workers=8,          
    dataloader_prefetch_factor=2,      
    learning_rate=2e-5, 
    lr_scheduler_type="cosine", 
    warmup_steps=0.05,          
    max_seq_length=1024,               
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("Starting Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Training Complete!")