import json
import csv

# Input and output files
input_file = "submission.json"
output_file = "submission.csv"

# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Write CSV with required format
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # Header
    writer.writerow(["item_index", "prediction"])

    # Rows
    for item in data:
        writer.writerow([
            item["item_index"],
            item["answer"]  # map answer -> prediction
        ])

print(f"CSV saved to {output_file}")