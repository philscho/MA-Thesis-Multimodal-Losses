import os
import json
import csv

# Paths
folder = "/home/phisch/multimodal/test_results/zero_shot"
csv_path = "/home/phisch/multimodal/test_results/model_scores_zero-shot.csv"

# Read CSV header
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)

# Prepare to append rows
rows_to_append = []

for fname in os.listdir(folder):
    if fname.endswith(".json") and not fname.startswith("ITM_only.json"):
        json_path = os.path.join(folder, fname)
        with open(json_path, "r") as f:
            data = json.load(f)

        # Model ID from filename (before first '-')
        model_id = fname.split('-')[0]

        # Try to get model_name from config
        model_name = ""
        try:
            model_name = data["config"]["checkpoints"][model_id]["model"]
        except Exception:
            pass

        # Try to get dataset_fraction from config
        dataset_fraction = ""
        try:
            dataset_fraction = data["config"]["checkpoints"][model_id]["subset_fraction"]
        except Exception:
            pass

        # Get templates info from config if available
        templates = data.get("config", {}).get("callbacks", {}).get("zeroshot", {}).get("callback", {}).get("templates", [])
        # Map template_count to template set index (3->0, 5->1, 9->2, 18->3, etc.)
        template_count_to_index = {len(tpl): idx for idx, tpl in enumerate(templates)}

        for result in data["results"]:
            dataset = result["dataset"]
            template_count = result.get("template_count", "")
            mode = result.get("mode", "")
            metrics = result.get("metrics", {})
            # Find which template set this result corresponds to (by template_count)
            method_notes = f"{template_count}_templates"
            for metric, score in metrics.items():
                row = [
                    model_id,
                    dataset,
                    "zeroshot",  # method
                    mode,
                    metric,
                    score,
                    method_notes,
                    "",  # dataset_notes
                    model_name,
                    dataset_fraction
                ]
                rows_to_append.append(row)

# Append new rows to CSV
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    for row in rows_to_append:
        writer.writerow(row)

print(f"Appended {len(rows_to_append)} rows to {csv_path}")