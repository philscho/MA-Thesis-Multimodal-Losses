import csv
import yaml

# csv_path = "test_results/model_scores_zero-shot.csv"
csv_path = "test_results/linear_probing_layers_results.csv"
yaml_path = "configs/checkpoints/model_id_mapping.yaml"
output_path = csv_path

# Load YAML mapping
with open(yaml_path, "r") as f:
    mapping = yaml.safe_load(f)

# Read CSV and update/add model_name and subset_fraction columns
rows = []
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames
    if "model_name" not in fieldnames:
        fieldnames.append("model_name")
    if "dataset_fraction" not in fieldnames:
        fieldnames.append("dataset_fraction")
    for row in reader:
        model_id = row["model_id"]
        # Only fill if missing or empty
        if "model_name" not in row or not row["model_name"]:
            row["model_name"] = mapping.get(model_id, {}).get("name", "")
        if "dataset_fraction" not in row or not row["dataset_fraction"]:
            row["dataset_fraction"] = mapping.get(model_id, {}).get("subset_fraction", "")
        rows.append(row)

# Write updated CSV
with open(output_path, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved updated CSV to {output_path}")