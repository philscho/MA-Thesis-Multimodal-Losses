#%%
import os
import torch
import pickle
import hydra
from omegaconf import OmegaConf
from torchvision import transforms
from lightning.pytorch import seed_everything
from tqdm import tqdm
import rootutils
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks.linear_probe_callback import train_linear_probe, eval_linear_probe, setup_linear_classifiers
from src.data.data_module import MyDataModule
from src.model.model_module import LitMML, MLMWrapper
from src.model.utils import get_model_and_processor

def ensure_dir_exists(path):
    """Ensure the parent directory of the given path exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def eval_linear_probe_per_layer(config, ckpt_path, dataset):
    device = torch.device(f'cuda:{config.lightning.trainer.devices[0]}')
    model, processor = get_model_and_processor(config.model)
    model = MLMWrapper(model)
    lit_model = LitMML.load_from_checkpoint(
        ckpt_path,
        model=model,
        processor=processor,
        map_location=device,
    )
    model, processor = lit_model.model, lit_model.processor

    data_module = MyDataModule(
        config.data,
        processor,
        augmentation=transforms.RandAugment(),
        num_views=2,
    )
    dataloaders = data_module.get_test_dataloaders()
    train_dataloader = dataloaders[dataset]["train"]
    test_dataloader = dataloaders[dataset]["test"]
    num_classes = train_dataloader.dataset.num_classes

    INTERESTING_LAYERS = {
        f"encoder_layer_{i}": model.basemodel.vision_model.encoder.layer[i] for i in range(12)
    }
    INTERESTING_LAYERS["projection_layer"] = model.basemodel.visual_projection

    layer_data_dict = {}
    for MAIN_LAYER_KEY in tqdm(INTERESTING_LAYERS, desc="Going through the layers"):
        def get_reps(x):
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    if isinstance(output, tuple):
                        activation[name] = output[0].detach()
                    else:
                        activation[name] = output.detach()
                return hook

            for k, v in INTERESTING_LAYERS.items():
                if k == MAIN_LAYER_KEY:
                    v.register_forward_hook(get_activation(k))

            activation = {}
            _ = model.get_image_features(pixel_values=x)
            return activation

        if MAIN_LAYER_KEY == "projection_layer":
            hidden_size = 512
            def custom_forward_func(x):
                outs = get_reps(x)[MAIN_LAYER_KEY]
                return outs
        else:
            hidden_size = 768
            def custom_forward_func(x):
                outs = get_reps(x)[MAIN_LAYER_KEY]
                return outs[:, 0, :]

        model.to(device)
        model.eval()
        print(f"{MAIN_LAYER_KEY}")

        classifiers, params = setup_linear_classifiers(
            out_dim=hidden_size, num_classes=num_classes, learning_rates=[0.0001]
        )

        trained_linear_probe = train_linear_probe(
            forward_func=custom_forward_func,
            dataloader=train_dataloader,
            dataset_name=dataset,
            classifiers=classifiers,
            parameters=params,
            max_epochs=500,
            verbose=True,
            device=device,
        )

        result = eval_linear_probe(
            forward_func=custom_forward_func,
            classifiers=trained_linear_probe,
            dataloader=test_dataloader,
            num_classes=num_classes,
            confusion_matrix=False,
            top_k=(1, 3, 5),
            verbose=True,
            device=device,
        )

        layer_data_dict[MAIN_LAYER_KEY] = result

    return layer_data_dict

@hydra.main(version_base=None, config_path="../../configs", config_name="linear_per_layer")
def main(config):
    print(OmegaConf.to_yaml(config))
    torch.set_float32_matmul_precision("medium")
    seed_everything(config.lightning.seed, workers=True)

    ckpt_list = config.checkpoints
    dataset = config.dataset

    print(f"Running for checkpoints: {ckpt_list}")
    for model_id, model_info in ckpt_list.items():
        print('--')
        print(model_id)
        print('--')
        if "path" in model_info:
            ckpt_path = model_info["path"]
        elif model_id == "n5trwpk9":
            ckpt_path = "/home/data/bhavin/n5trwpk9/ckpt-epoch=34-loss-val=4.378.ckpt"
        elif model_id == "zathvtrx":
            ckpt_path = "/data/bhavin/ckpts_old/zathvtrx/ckpt-epoch=14-loss-val=0.002.ckpt"
        else:
            ckpt_path = os.path.join(config.paths.ckpts_dir, model_id, "last.ckpt")

        data_dict = eval_linear_probe_per_layer(config, ckpt_path, dataset)
        out_path = os.path.join(ROOT, "test_results", f"{dataset}-{model_id}-linear_per_layer.p")
        ensure_dir_exists(out_path)
        try:
            with open(out_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Results written to: {os.path.abspath(out_path)}")
        except Exception as e:
            print(f"Failed to write file {out_path}: {e}")

if __name__ == '__main__':
    main()
