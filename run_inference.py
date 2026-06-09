import json
import csv
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH = "./lora-qwen-traffic-all-tasks" # Ensure this matches your new training output dir
TEST_JSON = "test/test.json"
OUTPUT_CSV = "submission.csv"
VIDEO_DIR = "test/videos/"

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
    video_path = f"{VIDEO_DIR}/{video_id}.mp4"
    
    system_instruction = "You are a traffic anomaly expert."
    if task_type == "temporal_localization":
        system_instruction += " Only output the final answer as a strict JSON object with 'start' and 'end' keys representing seconds. Example: {\"start\": 12.5, \"end\": 45.0}. Do not include reasoning."
    elif task_type in ["bcq", "bcq_openended"]:
        system_instruction += " Answer strictly with 'Yes' or 'No'."

    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                # Matched inference FPS to training FPS for consistency
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
        # Increased max_new_tokens to 512 to prevent open-ended text tasks from truncating
        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Storing raw text response exactly as Qwen output it
    submissions.append({
        "item_index": item["item_index"],
        "prediction": response
    })
    print(f"Processed item {item['item_index']} - Answer snippet: {response[:50]}...")

# Outputting to standard 2-column CSV Format
print(f"Writing to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["item_index", "prediction"])
    for sub in submissions:
        # The python csv writer automatically handles multi-line text wraps and quotes
        writer.writerow([sub["item_index"], sub["prediction"]])

print(f"Saved submission strictly to {OUTPUT_CSV}")