from datetime import datetime
import pickle
import hydra
from omegaconf import OmegaConf
import os
from pathlib import Path
import rootutils
import torch
from tqdm import tqdm
ROOT = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

from src.callbacks.zeroshot_callback import make_batches
from src.analysis.representations import (
    get_average_class_representations, 
    get_class_representations,
    compute_cka_and_kernel_matrices,
    get_mean_class_representations_per_layer
)
from src.data.data_module import MyDataModule
from src.model.model_module import LitMML, MLMWrapper
from src.model.utils import get_model_and_processor


def create_zero_shot_classifier(
    forward_func,
    classnames: list,
    templates: list = None,
    tokenizer=None,
    batch_size: int = 64,
    device: str = "cuda",
    verbose: bool = False,
    layers: list = None,  # NEW: list of layers to extract
):
    templates = ['{}'] if templates is None else templates
    if isinstance(templates, str):
        templates = [templates]
    num_templates = len(templates)
    batch_size = 2 ** ((batch_size // num_templates) - 1).bit_length()

    batch_class_names = make_batches(classnames, batch_size)
    zeroshot_weights_per_layer = {layer: [] for layer in layers}

    with torch.no_grad():
        bar = tqdm(batch_class_names, desc="Classifier weights...") if verbose else batch_class_names
        for batch_class_name in bar:
            texts = [template.format(classname) for classname in batch_class_name for template in templates]
            if tokenizer is not None:
                input = tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = input['input_ids'].to(device)
                mask = input['attention_mask'].to(device)
            else:
                input_ids, mask = texts, None

            # Get all hidden states in one forward pass
            outputs = forward_func(input_ids, mask, output_hidden_states=True)
            hidden_states = outputs["hidden_states"]  # tuple: (embeddings, layer1, ..., last)

            for layer in layers:
                if layer == "last" or layer == -1:
                    rep = hidden_states[-1][:, 0, :]
                elif isinstance(layer, int):
                    rep = hidden_states[layer + 1][:, 0, :]  # +1 because 0 is embeddings
                else:
                    raise ValueError(f"Unknown layer: {layer}")

                rep = rep / rep.norm(dim=-1, keepdim=True)
                rep = rep.view(len(batch_class_name), num_templates, -1)
                rep_mean = rep.mean(dim=1)
                rep_mean = rep_mean / rep_mean.norm(dim=1).view(-1, 1)
                zeroshot_weights_per_layer[layer].append(rep_mean)

    # Stack and transpose for each layer
    for layer in zeroshot_weights_per_layer:
        zeroshot_weights_per_layer[layer] = torch.cat(zeroshot_weights_per_layer[layer], dim=0).to(device)

    return zeroshot_weights_per_layer


@hydra.main(version_base=None, config_path="../../configs", config_name="similarity_analysis")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    CKPT_DIR = Path(cfg.paths.ckpts_dir)
    CKPT_CONFIG_DIR = os.path.join(ROOT, "checkpoints")
    SAVE_DATA_DIR = os.path.join(ROOT, "data")
    SAVE_FIGURES_DIR = os.path.join(ROOT, "plots")

    MODEL_NAMES = ["SimCLR", "SimCLR + ITM", "CLIP + SimCLR", "CLIP + SimCLR + ITM", "CLIP + ITM", "CLIP"]
    DATASET_SIZES = ["full_dataset_im384", "full_dataset", "0.8_dataset", "0.6_dataset", "0.4_dataset", "0.2_dataset", "0.1_dataset", "0.05_dataset"]

    ## Computing representations
    ### Checkpoints and dataset config
    DATASET_SPLIT = cfg.dataset_split
    print("Dataset split: ", DATASET_SPLIT)

    INTERESTING_CHECKPOINTS = cfg.checkpoints
    MODEL_NAMES = [INTERESTING_CHECKPOINTS[ckpt]["model"] for ckpt in INTERESTING_CHECKPOINTS]
    print("Models: ", MODEL_NAMES)

    DATASET = cfg.dataset
    print("Dataset: ", DATASET)

    print(INTERESTING_CHECKPOINTS)

    ### Compute representations
    #### Init model and datasets
    model, processor = get_model_and_processor(cfg.model)
    model = MLMWrapper(model)
    
    data_module = MyDataModule(
        cfg.data,
        processor,
    )
    callback_dataloaders = data_module.get_test_dataloaders()
    dataloader = callback_dataloaders[DATASET]["test"]
    num_classes = dataloader.dataset.num_classes
    class_names = dataloader.dataset.classnames
    print(num_classes, class_names)
    
    #### Compute
    device = f"cuda:{cfg.device[0]}"
    print(f"Getting representations for dataset split {DATASET_SPLIT} models {MODEL_NAMES}")

    MEAN_CLASS_REPS = {}
    CKA_ALL_REPS = {}
    save_file_dir = os.path.join(SAVE_DATA_DIR, "representations", DATASET, DATASET_SPLIT)

    all_templates = cfg.callbacks.zeroshot.callback.templates

    for ckpt in INTERESTING_CHECKPOINTS:
        model_name = INTERESTING_CHECKPOINTS[ckpt]["model"]
        ckpt_path = INTERESTING_CHECKPOINTS[ckpt]["path"]
        # if ckpt == "n5trwpk9":
        #     ckpt_path = "/home/data/bhavin/n5trwpk9/ckpt-epoch=34-loss-val=4.378.ckpt"
        # elif ckpt == "zathvtrx":
        #     ckpt_path = "/data/bhavin/ckpts_old/zathvtrx/ckpt-epoch=14-loss-val=0.002.ckpt"
        # else:
        #     files = os.listdir(os.path.join(CKPT_DIR, ckpt))
        #     if len(files) > 1 and "last.ckpt" in files:
        #         files.remove("last.ckpt")
        #     ckpt_path = os.path.join(CKPT_DIR, ckpt, files[0])
        
        print(f"Loading model {ckpt}:{model_name} from checkpoint", ckpt_path)
        lit_model = LitMML.load_from_checkpoint(ckpt_path, model=model, processor=processor, map_location=device)
        model, processor = lit_model.model, lit_model.processor
        del lit_model
        torch.cuda.empty_cache()

        print("Computing class representations")
        layers_to_extract = getattr(cfg, "model_layers", ["last"])

        if cfg.get("visual_reps", False):
            forward_func = lambda x, **kwargs: model.basemodel.vision_model(pixel_values=x, output_hidden_states=True, **kwargs)
            class_representations_images_dict = get_mean_class_representations_per_layer(
                forward_func=forward_func,
                dataloader=dataloader,
                num_classes=num_classes,
                layers=layers_to_extract,
                device=device,
            )
            representations = []
            for layer, class_representations_images in class_representations_images_dict.items():
                class_representations_images = class_representations_images
                dictionary = {
                    "modality": "vision",
                    "layer": layer,
                    "data": class_representations_images,
                    "embedding_dim": class_representations_images.shape[1],
                    "dtype": str(class_representations_images.dtype),
                    "created": datetime.now().strftime("%Y-%m-%d"),
                    "notes": "",
                }
                representations.append(dictionary)

            save_file_path = os.path.join(
                save_file_dir, f"{DATASET}-{ckpt}-visual_representations.pkl"
            )
            if os.path.exists(save_file_path):
                with open(save_file_path, "rb") as f:
                    data = pickle.load(f)
                if "representations" in data and isinstance(data["representations"], list):
                    data["representations"].extend(representations)
                else:
                    data["representations"] = representations
            else:
                data = {
                    "dataset": DATASET,
                    "model": ckpt,
                    "model_name": model_name,
                    "class_names": class_names,
                    "representations": representations
                }
            with open(save_file_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved: {save_file_path}")

        if cfg.get("text_reps", False):
            for templates in all_templates:
                # Pass layers to _create_zero_shot_classifier
                class_representations_texts_dict = create_zero_shot_classifier(
                    forward_func=lambda x, y, **kwargs: model.basemodel.text_model(input_ids=x, attention_mask=y, **kwargs),
                    tokenizer=processor,
                    classnames=class_names,
                    templates=templates,
                    device=device,
                    layers=layers_to_extract,
                )

                for layer, class_representations_texts in class_representations_texts_dict.items():
                    class_representations_texts = class_representations_texts.detach().cpu().numpy()
                    dictionary = {
                        "modality": "text",
                        "layer": layer,
                        "templates": templates,
                        "data": class_representations_texts,
                        "embedding_dim": class_representations_texts.shape[1],
                        "dtype": str(class_representations_texts.dtype),
                        "created": datetime.now().strftime("%Y-%m-%d"),
                        "notes": "",
                    }
                    save_file_path = os.path.join(
                        save_file_dir, f"{DATASET}-{ckpt}-text_representations.pkl"
                    )
                    if os.path.exists(save_file_path):
                        with open(save_file_path, "rb") as f:
                            data = pickle.load(f)
                        if "representations" in data and isinstance(data["representations"], list):
                            data["representations"].append(dictionary)
                        else:
                            data["representations"] = [dictionary]
                    else:
                        data = {
                            "dataset": DATASET,
                            "model": ckpt,
                            "model_name": model_name,
                            "class_names": class_names,
                            "representations": [dictionary]
                        }
                    with open(save_file_path, "wb") as f:
                        pickle.dump(data, f)
                    print(f"Saved: {save_file_path}")
        
        # print("Getting class representations")
        # save_file_path = os.path.join(save_file_dir, f"{DATASET}-{ckpt}-mean_class_reps.pkl")
        
        # # Save all representations for CKA
        # mean_vec = (
        #     torch.concatenate([torch.stack(CLASS_REPS[x]) for x in CLASS_REPS], axis=0)
        #     .cpu()
        #     .numpy()
        # )
        # CKA_ALL_REPS[ckpt] = mean_vec

        # del CLASS_REPS
        # torch.cuda.empty_cache()

    # save_file_path = os.path.join(save_file_dir, f"{DATASET}-cka_and_kernel_matrices.pkl")
    # CKA_MAT, KERNEL_MAT = compute_cka_and_kernel_matrices(CKA_ALL_REPS, device=device, save_file_path=save_file_path)


if __name__ == "__main__":
    main()
