import json
import os
import random

# All 10 tasks from the NVIDIA PhysicalAI-Traffic-Anomaly-Reasoning dataset
TRAIN_FILES = [
    "bcq.json", "bcq_openended.json", "mcq.json", "mcq_openended.json",
    "open_qa.json", "scene_description.json", "video_summarization.json",
    "temporal_localization.json", "causal_linkage.json", "temporal_description.json"
]

TRAIN_DIR = "/media/RAID5Array/backup_home/tindd4/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/train"
OUTPUT_FILE = "all_tasks_merged.json"
TARGET_SAMPLES = 2000  # Adjust this baseline target based on your hardware/time constraints

def process_and_merge():
    task_data_dict = {}

    for file_name in TRAIN_FILES:
        file_path = os.path.join(TRAIN_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found in {TRAIN_DIR}. Skipping.")
            continue
            
        task_type = file_name.replace(".json", "")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Robust JSON Parsing
        if isinstance(data, dict):
            if "items" in data:
                data_list = data["items"]
            else:
                data_list = list(data.values())
        elif isinstance(data, list):
            data_list = data
        else:
            print(f"Error: Unknown JSON structure in {file_name}")
            continue
            
        for item in data_list:
            item["task_type"] = task_type    
            if task_type not in task_data_dict:
                task_data_dict[task_type] = []
            task_data_dict[task_type].append(item)

    merged_data = []
    
    # Balance and oversample the dataset
    for task_type, items in task_data_dict.items():
        if len(items) > TARGET_SAMPLES:
            # Cap overly large tasks to prevent domination
            sampled_items = random.sample(items, TARGET_SAMPLES)
        else:
            # Upsample smaller tasks (e.g., temporal localization)
            sampled_items = random.choices(items, k=TARGET_SAMPLES)
            
        # Give BCQ and MCQ tasks a 1.5x weight advantage
        if task_type in ["bcq", "mcq", "bcq_openended", "mcq_openended"]:
            sampled_items.extend(random.choices(items, k=int(TARGET_SAMPLES * 0.5)))
            
        merged_data.extend(sampled_items)
        print(f"Processed {task_type}: Kept {len(sampled_items)} items (Balanced).")

    # Shuffle to prevent catastrophic forgetting
    random.seed(42)
    random.shuffle(merged_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"\nSuccessfully merged {len(merged_data)} total items into {OUTPUT_FILE}")

if __name__ == "__main__":
    process_and_merge()