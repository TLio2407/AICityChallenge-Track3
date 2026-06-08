import json
import os
import random

# All 10 tasks from the NVIDIA PhysicalAI-Traffic-Anomaly-Reasoning dataset
TRAIN_FILES = [
    "bcq.json", "bcq_openended.json", "mcq.json", "mcq_openended.json",
    "open_qa.json", "scene_description.json", "video_summarization.json",
    "temporal_localization.json", "causal_linkage.json", "temporal_description.json"
]

TRAIN_DIR = "train/" # Adjust if your path is different
OUTPUT_FILE = "train/all_tasks_merged.json"

def process_and_merge():
    merged_data = []
    stats = {}

    for file_name in TRAIN_FILES:
        file_path = os.path.join(TRAIN_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found in {TRAIN_DIR}. Skipping.")
            continue
            
        task_type = file_name.replace(".json", "")
        
        with open(file_path, "r") as f:
            data = json.load(f)
            
        kept_items = 0
        
        for item in data:
            # Tag the item with its task type so the trainer knows how to prompt it
            item["task_type"] = task_type
            
            # Special cleaning rule for the noisy temporal localization data
            if task_type == "temporal_localization":
                try:
                    answer_json = json.loads(item["answer"])
                    start_time = float(answer_json.get("start", 0))
                    end_time = float(answer_json.get("end", 0))
                    # Keep only mathematically valid timestamps under 5 minutes
                    if (start_time < end_time) and (end_time <= 300) and (end_time - start_time > 0.5):
                        merged_data.append(item)
                        kept_items += 1
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue # Drop malformed temporal data
            else:
                # For all other 9 tasks, keep the data as-is
                merged_data.append(item)
                kept_items += 1
                
        stats[task_type] = kept_items
        print(f"Processed {task_type}: Kept {kept_items} items.")

    # Shuffle the dataset to prevent catastrophic forgetting during multi-task learning
    random.seed(42)
    random.shuffle(merged_data)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"\nSuccessfully merged {len(merged_data)} total items into {OUTPUT_FILE}")

if __name__ == "__main__":
    process_and_merge()