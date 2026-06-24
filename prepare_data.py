import json
import os
import random

# All 10 tasks from the NVIDIA PhysicalAI-Traffic-Anomaly-Reasoning dataset
TRAIN_FILES = [
    "bcq.json", "bcq_openended.json", "mcq.json", "mcq_openended.json",
    "open_qa.json", "scene_description.json", "video_summarization.json",
    "temporal_localization.json", "causal_linkage.json", "temporal_description.json"
]

TRAIN_DIR = "/media/RAID5Array/haolp/AIC26/PhysicalAI-Traffic-Anomaly-Reasoning/train"
OUTPUT_FILE = "all_tasks_merged.json"

# PRIORITY UPSAMPLING WEIGHTS
TASK_WEIGHTS = {
    "bcq": 3,
    "mcq": 3,
    "bcq_openended": 2,
    "mcq_openended": 2
}

def process_and_merge():
    merged_data = []
    stats = {}

    for file_name in TRAIN_FILES:
        file_path = os.path.join(TRAIN_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found in {TRAIN_DIR}. Skipping.")
            continue
            
        task_type = file_name.replace(".json", "")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # --- ROBUST JSON PARSING ---
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
            
        kept_items = 0
        weight = TASK_WEIGHTS.get(task_type, 1) # Default to 1 for non-priority tasks
        
        for item in data_list:
            item["task_type"] = task_type    
            
            # Upsampling loop based on task importance
            for _ in range(weight):
                merged_data.append(item.copy())
            kept_items += weight
                
        stats[task_type] = kept_items
        print(f"Processed {task_type}: Generated {kept_items} instances (Weight: {weight}).")

    # Shuffle to prevent catastrophic forgetting across batches
    random.seed(42)
    random.shuffle(merged_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"\nSuccessfully merged {len(merged_data)} items into {OUTPUT_FILE}")
    print("Final Task Distribution:", stats)

if __name__ == "__main__":
    process_and_merge()