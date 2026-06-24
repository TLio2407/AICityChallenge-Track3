import os
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

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
        
        if task_type == "bcq":
            sys_prompt += " Answer strictly with 'Yes' or 'No' and nothing else."
        elif task_type == "bcq_openended":
            sys_prompt += " Answer with 'Yes' or 'No' first, followed by a detailed reasoning explanation."
        elif task_type == "mcq":
            sys_prompt += " Select the best option. Output only the letter (A, B, C, or D) of the correct answer."
        elif task_type == "mcq_openended":
            sys_prompt += " Select the best option and explain your reasoning in detail."
        elif task_type == "temporal_localization":
            sys_prompt += " Provide exact temporal boundaries in strict JSON format: {'start': X, 'end': Y}."
        else:
            sys_prompt += " Analyze the video and answer the question in detail."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": f"/media/RAID5Array/haolp/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/videos/{video_path}"},
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

print("Loading Qwen2-VL Model and Processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# FIX: Set padding side explicitly for the SFTTrainer warning
processor.tokenizer.padding_side = "right"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# FIX: Disable use_cache to avoid conflicts with gradient checkpointing
model.config.use_cache = False

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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, 
    learning_rate=2e-5, 
    lr_scheduler_type="cosine", 
    warmup_steps=1,          
    max_seq_length=2048, 
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    dataset_text_field="text",
    
    # FIX: Explicitly handle the PyTorch use_reentrant warning
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    # FIX: Use processing_class instead of relying on deprecated fallback mechanisms
    processing_class=processor.tokenizer 
)

print("Starting Training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Training Complete!")