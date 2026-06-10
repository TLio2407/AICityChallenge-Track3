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

def process_and_merge():
    merged_data = []

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
            
        task_items = []
        for item in data_list:
            item["task_type"] = task_type    
            task_items.append(item)
            
        # Keep ALL original items
        sampled_items = task_items.copy()
            
        # Give BCQ and MCQ tasks a weight advantage to prioritize their accuracy
        # by oversampling them by an additional 50%
        if task_type in ["bcq", "mcq", "bcq_openended", "mcq_openended"]:
            extra_samples = int(len(task_items) * 0.5)
            sampled_items.extend(random.choices(task_items, k=extra_samples))
            print(f"Processed {task_type}: Kept all {len(task_items)} items + {extra_samples} oversampled (Weight Boost).")
        else:
            print(f"Processed {task_type}: Kept all {len(task_items)} items.")
            
        merged_data.extend(sampled_items)

    # Shuffle everything to prevent catastrophic forgetting during training
    random.seed(42)
    random.shuffle(merged_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"\nSuccessfully merged {len(merged_data)} total items into {OUTPUT_FILE}")

if __name__ == "__main__":
    process_and_merge()