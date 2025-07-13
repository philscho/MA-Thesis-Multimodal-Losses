import pickle
import csv
import os

# Paths
folder = "/home/phisch/multimodal/test_results/linear_probe/mlm-0.05_0.4"
csv_path = "/home/phisch/multimodal/test_results/model_scores_linear_probe.csv"

# Read CSV header
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)

rows_to_append = []

for fname in os.listdir(folder):
    if not fname.endswith("-linear_probe-results.p"):
        continue
    model_id = fname.split("-")[0]
    pkl_path = os.path.join(folder, fname)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    config = data.get("__config__", {})
    model_info = config.get("checkpoints", {}).get(model_id, {})
    model_name = model_info.get("model", "")
    dataset_fraction = model_info.get("subset_fraction", "")
    method_notes = "last_image_layer"

    for key in data:
        if not key.startswith("linear_"):
            continue
        dataset = key.replace("linear_", "")
        metrics = data[key]
        for metric in ["Top1Accuracy", "Top3Accuracy", "Top5Accuracy"]:
            score = metrics.get(metric, "")
            row = [
                model_id,
                dataset,
                "linear",  # method
                metric,
                score,
                method_notes,  # method_notes
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