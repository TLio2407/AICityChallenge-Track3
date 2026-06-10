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
    
    # Must exactly match the strict instructions from the training script
    system_instruction = "You are a traffic anomaly expert."
    if task_type == "temporal_localization":
        system_instruction += " Output the final answer as a strict JSON: {\"start\": X.X, \"end\": Y.Y}. No other text."
    elif task_type in ["bcq", "bcq_openended"]:
        system_instruction += " You must answer strictly with a single word: 'Yes' or 'No'. Do not explain."
    elif task_type in ["mcq", "mcq_openended"]:
        system_instruction += " Answer strictly with the exact option provided. Do not include reasoning."
    else:
        system_instruction += " Analyze the video and answer the question in detail."

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
        # Enforcing purely deterministic generation for evaluation stability
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=False,          # Forces greedy decoding, ignoring temperature
            repetition_penalty=1.05   # Helps prevent looping on open-ended tasks
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    submissions.append({
        "item_index": item["item_index"],
        "prediction": response.strip() # Ensure cleanly stripped predictions for the CSV
    })
    print(f"Processed item {item['item_index']} - Answer snippet: {response[:50]}...")

print(f"Writing to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["item_index", "prediction"])
    for sub in submissions:
        writer.writerow([sub["item_index"], sub["prediction"]])

print(f"Saved submission to {OUTPUT_CSV}")