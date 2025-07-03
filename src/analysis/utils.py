import os
from omegaconf import OmegaConf
import pickle
from pathlib import Path
import rootutils

ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)
CKPT_CONFIG_DIR = ROOT / "configs" / "checkpoints"

DATASET_SIZES = ["0.2_dataset", "0.4_dataset", "0.6_dataset", "0.8_dataset", "full_dataset", "full_dataset_im384"]
DICT_CKPTS_CONFIG = {size: OmegaConf.load(CKPT_CONFIG_DIR / f"{size}.yaml") for size in DATASET_SIZES}

def load_zeroshot_results(
        data_dir: str = "/home/phisch/multimodal/test_results"
    ) -> dict:
    
    data_dir = Path(data_dir)
    all_data = {"0.2_dataset": {}, 
                "0.4_dataset": {}, 
                "0.6_dataset": {}, 
                "0.8_dataset": {}, 
                "full_dataset": {}, 
                "full_dataset_im384": {}
                }
    
    for dataset_folder in os.listdir(data_dir):
        dataset_folder = data_dir / dataset_folder
        for file_name in os.listdir(dataset_folder):
            if "-zeroshot-results" in file_name:
                model_id = file_name.split("-")[0]
                with open(dataset_folder / file_name, "rb") as f:
                    all_data[dataset_folder.name][model_id] = pickle.load(f)
    
    return all_data

def load_linear_probe_results(
        data_dir: str = "/home/phisch/multimodal/test_results",
        sub_dir: str = None
    ) -> dict:
    
    sizes = ["0.2_dataset", "0.4_dataset", "0.6_dataset", "0.8_dataset", "full_dataset", "full_dataset_im384"]

    data_dir = Path(data_dir)
    all_data = {size: {} for size in sizes}
    
    for size in sizes:
        dataset_folder = data_dir / size / sub_dir if sub_dir else data_dir / size
        for file_name in os.listdir(dataset_folder):
            if "-linear_probe-results" in file_name:
                model_id = file_name.split("-")[0]
                with open(dataset_folder / file_name, "rb") as f:
                    all_data[size][model_id] = pickle.load(f)    
    
    return all_data

def load_cka_results(data_dir: str) -> dict:
    data_dir = Path(data_dir)
    cka_all_splits = {}

    for ds_size in os.listdir(data_dir):
        ds_size_dir = os.path.join(data_dir, ds_size)
        for file_name in os.listdir(ds_size_dir):
            if "cka_and_kernel_matrices" in file_name:
                file_path = os.path.join(ds_size_dir, file_name)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                cka_all_splits[ds_size] = data
    
    return cka_all_splits

def load_linear_probe_across_layers_results(data_dir: str, dataset_name) -> dict:
    data_dir = Path(data_dir)
    all_data = {"0.05_dataset": {}, 
                "0.1_dataset": {}, 
                "0.2_dataset": {}, 
                "0.4_dataset": {}, 
                "0.6_dataset": {}, 
                "0.8_dataset": {}, 
                "full_dataset": {}, 
                "full_dataset_im384": {},
                "full_dataset_aug": {},
                }
    
    for dataset_folder in os.listdir(data_dir):
        if dataset_folder not in ["0.05_dataset", "0.1_dataset", "0.2_dataset", "0.4_dataset", "0.6_dataset", "0.8_dataset", "full_dataset", "full_dataset_im384", "full_dataset_aug"]: continue
        dataset_folder = data_dir / dataset_folder
        linear_folder = dataset_folder / "linear_per_layer"
        print(linear_folder)
        for file_name in os.listdir(linear_folder):
            if f"{dataset_name}" in file_name:
                model_id = file_name.split("-")[1]
                with open(linear_folder / file_name, "rb") as f:
                    all_data[dataset_folder.name][model_id] = pickle.load(f)
    
    return all_data