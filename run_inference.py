import json
import csv
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH = "./lora-qwen-traffic-all-tasks"
TEST_JSON = "/media/RAID5Array/backup_home/tindd4/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/test/test.json"
OUTPUT_CSV = "submission.csv"
VIDEO_DIR = "/media/RAID5Array/backup_home/tindd4/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/test/videos/"

print("Loading Base VLM and LoRA Adapter...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

with open(TEST_JSON, "r", encoding="utf-8") as f:
    test_data = json.load(f)

submissions = []

print("Running Video Inference...")
for item in test_data["items"]:
    video_id = item["video_id"]
    task_type = item.get("task_type", "open_qa")
    video_path = f"{VIDEO_DIR}/{video_id}"
    
    # Enhanced Persona and Task-Specific Prompting
    system_instruction = "You are an expert AI system specialized in traffic anomaly detection and reasoning. Your task is to analyze the provided traffic video and respond based on the specific constraints."
    
    # Set dynamic generation params based on task strictness
    gen_kwargs = {"max_new_tokens": 512}
    
    if task_type == "temporal_localization":
        system_instruction += " Analyze the video to identify the exact onset and conclusion of the anomaly. Provide your answer strictly as a JSON object with 'start' and 'end' keys representing seconds. Example: {\"start\": 12.5, \"end\": 45.0}. Do NOT include any other text, reasoning, or markdown formatting."
        gen_kwargs.update({"do_sample": False}) # Greedy decoding for exact formatting
        
    elif task_type in ["bcq", "bcq_openended"]:
        system_instruction += " Answer strictly with 'Yes' or 'No' based solely on the visual evidence."
        gen_kwargs.update({"do_sample": False})
        
    elif task_type in ["mcq", "mcq_openended"]:
        system_instruction += " Select the best option. Answer concisely by providing the exact text of the correct option."
        gen_kwargs.update({"do_sample": False})
        
    else:
        # Chain-of-Thought for Open QA
        system_instruction += " Think step-by-step: first identify the vehicles or pedestrians involved, describe their actions chronologically, and finally deduce the anomaly to answer the question in detail."
        gen_kwargs.update({"do_sample": True, "temperature": 0.2, "top_p": 0.9})

    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 2.0}, 
                {"type": "text", "text": item["question"]}
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    submissions.append({
        "item_index": item["item_index"],
        "prediction": response
    })
    print(f"Processed item {item['item_index']} [{task_type}] - Answer snippet: {response[:50]}...")

print(f"Writing to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["item_index", "prediction"])
    for sub in submissions:
        writer.writerow([sub["item_index"], sub["prediction"]])

print(f"Saved submission strictly to {OUTPUT_CSV}")