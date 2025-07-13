import yaml
import os

yaml_path = "/home/phisch/multimodal/configs/checkpoints/mlm-0.05_0.4.yaml"
checkpoints_root = "/home/data/bhavin"  # <-- change this to your checkpoints folder

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

for model_id in data.keys():
    # Skip comments or non-model entries
    if not isinstance(data[model_id], dict):
        continue
    model_path = os.path.join(checkpoints_root, model_id, "last.ckpt")
    data[model_id]["path"] = model_path

# Write back to YAML
with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)