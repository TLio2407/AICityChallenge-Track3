import json
import csv
import torch
from peft import PeftModel
# UPDATED: Import the Qwen3 architecture class
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# UPDATED: Match the Qwen3-VL Model ID from the training script
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
# UPDATED: Match the new output directory from the training script
ADAPTER_PATH = "./lora-qwen3-traffic-all-tasks"
TEST_JSON = "/media/RAID5Array/haolp/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/test/test.json"
OUTPUT_CSV = "submission.csv"
VIDEO_DIR = "/media/RAID5Array/haolp/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/test/videos/"

print("Loading Base VLM and LoRA Adapter...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# UPDATED: Use the Qwen3 model class for instantiation
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
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
    question = item.get("question", "")
    video_path = f"{VIDEO_DIR}/{video_id}"
    
    # Mirroring the training prompts exactly
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
            {"type": "video", "video": video_path},
            {"type": "text", "text": question}
        ]}
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

    # Dynamic Generation Parameters
    if task_type in ["bcq", "mcq"]:
        temp = 0.001 
        max_tok = 10 
        do_sample_flag = False # Force greedy decoding
    else:
        temp = 0.2
        max_tok = 512
        do_sample_flag = True

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tok, 
            temperature=temp,
            do_sample=do_sample_flag
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    submissions.append({
        "item_index": item["item_index"],
        "prediction": response.strip() # Strip added to ensure trailing spaces don't break evaluation
    })
    print(f"Processed {item['item_index']} [{task_type}] - Ans: {response[:50].strip()}...")

# Outputting to standard 2-column CSV Format
print(f"Writing to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["item_index", "prediction"])
    for sub in submissions:
        writer.writerow([sub["item_index"], sub["prediction"]])
print("Inference Complete!")