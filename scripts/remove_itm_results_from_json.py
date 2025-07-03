import os
import json

folder = "/home/phisch/multimodal/test_results/full_dataset_aug_mlm/template_sets_and_itm"

for fname in os.listdir(folder):
    if fname.endswith(".json"):
        path = os.path.join(folder, fname)
        with open(path, "r") as f:
            data = json.load(f)
        if "results" in data:
            original_len = len(data["results"])
            # Remove all entries where mode == "itm"
            data["results"] = [r for r in data["results"] if r.get("mode") != "itm"]
            print(f"{fname}: removed {original_len - len(data['results'])} itm results")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)