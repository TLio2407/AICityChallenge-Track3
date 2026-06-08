import json
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH = "./lora-qwen-traffic-reasoning"
TEST_JSON = "test/test.json"
OUTPUT_JSON = "submission.json"

print("Loading Base VLM and LoRA Adapter...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

with open(TEST_JSON, "r") as f:
    test_data = json.load(f)

submissions = []

print("Running Video Inference...")
for item in test_data["items"]:
    video_id = item["video_id"]
    task_type = item.get("task_type", "open_qa")
    video_path = f"test/videos/{video_id}.mp4"
    
    # Task-specific prompt engineering
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
                {"type": "video", "video": video_path, "fps": 1.0}, # Extract 1 frame per second
                {"type": "text", "text": item["question"]}
            ],
        }
    ]

    # Process vision inputs specifically for Qwen2-VL
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
        generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
    
    # Isolate the newly generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    submissions.append({
        "item_index": item["item_index"],
        "video_id": video_id,
        "answer": response.strip()
    })
    print(f"Processed item {item['item_index']} - Answer: {response.strip()}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(submissions, f, indent=4)
print(f"Saved submission to {OUTPUT_JSON}")