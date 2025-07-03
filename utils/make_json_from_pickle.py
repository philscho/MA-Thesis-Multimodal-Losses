import pickle
import json
import os
from glob import glob

input_folder = '/home/phisch/multimodal/test_results/0.1_dataset'
output_folder = input_folder  # or set to another folder if you want

for pkl_path in glob(os.path.join(input_folder, '*.p')):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    results = []
    config = None

    # Extract config if present
    if '__config__' in data:
        config = data['__config__']
        # print(type(config))
        config = OmegaConf.to_container(config, resolve=False)
        # Convert OmegaConf config to dict if needed
        try:
            from omegaconf import OmegaConf, DictConfig
            # if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        except Exception:
            pass
        del data['__config__']

    for key in data:
        if not key.startswith('zeroshot_'):
            continue
        dataset = key.replace('zeroshot_', '')
        for template_count_str, metrics_dict in data[key].items():
            template_count = int(template_count_str.split('_')[0])
            # Check if metrics are split by mode
            if any(m in metrics_dict for m in ['regular', 'itm']):
                for mode in ['regular', 'itm']:
                    if mode in metrics_dict:
                        metrics = {}
                        for acc_key in ['Top1Accuracy', 'Top3Accuracy', 'Top5Accuracy']:
                            if acc_key in metrics_dict[mode]:
                                metrics[acc_key] = float(metrics_dict[mode][acc_key])
                        results.append({
                            "dataset": dataset,
                            "template_count": template_count,
                            "mode": mode,
                            "metrics": metrics
                        })
            else:
                # Only one mode present, assume 'regular'
                metrics = {}
                for acc_key in ['Top1Accuracy', 'Top3Accuracy', 'Top5Accuracy']:
                    if acc_key in metrics_dict:
                        metrics[acc_key] = float(metrics_dict[acc_key])
                results.append({
                    "dataset": dataset,
                    "template_count": template_count,
                    "mode": "regular",
                    "metrics": metrics
                })

    json_data = {"results": results}
    if config is not None:
        json_data["config"] = config

    json_path = os.path.splitext(pkl_path)[0] + '-zeroshot-all-.json'
    with open(json_path, 'w') as jf:
        json.dump(json_data, jf, indent=2)