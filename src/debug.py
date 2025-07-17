import pickle
import hydra
import lightning as pl
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import rootutils
import torch
import os
from omegaconf import OmegaConf
from pathlib import Path
import wandb
import json

from src.callbacks.utils import instantiate_linear_probe_callbacks, instantiate_zeroshot_callbacks
from src.data.data_module import MyDataModule
from src.model.model_module import LitMML
from src.model.utils import get_model_and_processor

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.analysis.eval_model import get_model_data

@hydra.main(version_base=None, config_path="../configs", config_name="debug")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("medium")
    seed_everything(cfg.lightning.seed, workers=True)

    # if rank_zero_only.rank == 0:
    #     cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    #     # wandb_logger.experiment.config.update(cfg_dict)

    if "different_modality_weights" in cfg:
        ckpt_path_2 = cfg.different_modality_weights.ckpt_path
    else:
        ckpt_path_2 = None
    
    for model_id, model_info in cfg.checkpoints.items():
        wandb_config = OmegaConf.to_container(cfg.logger.wandb, resolve=True)
        wandb_config = OmegaConf.create(wandb_config)
        wandb_config.tags.append(model_id)

        if "path" in model_info:
            ckpt_path = model_info["path"]
        elif model_id == "n5trwpk9":
            ckpt_path = "/home/data/bhavin/n5trwpk9/ckpt-epoch=34-loss-val=4.378.ckpt"
        elif model_id == "zathvtrx":
            ckpt_path = "/data/bhavin/ckpts_old/zathvtrx/ckpt-epoch=14-loss-val=0.002.ckpt"
        else:
            ckpt_dir = os.path.join(cfg.paths.ckpts_dir, model_id)
            ckpt_files = os.listdir(ckpt_dir)
            if len(ckpt_files) > 1 and "last.ckpt" in ckpt_files:
                ckpt_files.remove("last.ckpt")
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])

        device_idx = cfg.lightning.trainer.devices[0]
    
        model, processor = get_model_and_processor(cfg.model)
        # if "0.1_and_0.5_ckpts" in ckpt_path or "higher_augmentations_ckpts" in ckpt_path:
        #     model = MLMWrapper(model)
        # model = MLMWrapper(model)
        lit_model = LitMML.load_from_checkpoint(ckpt_path, 
                                                model=model, 
                                                processor=processor,
                                                map_location=f"cuda:{device_idx}",
                                                )
        model, processor = lit_model.model, lit_model.processor

        data_module = MyDataModule(
            cfg.data,
            processor,
            # augmentation=transforms.RandAugment(),
            # num_views=2,
        )
        dataloaders = data_module.get_test_dataloaders()

        callbacks = dict()
        # if "zeroshot" in callbacks_config:
        if cfg.callbacks.get("zeroshot", None):
            if cfg.callbacks.zeroshot.use_itm_head:
                itm_head = getattr(lit_model, 'itm_head', None)
            else: itm_head = None
            zeroshot_callbacks = instantiate_zeroshot_callbacks(
                    zeroshot_config=cfg.callbacks.zeroshot.callback,
                    dataloaders=dataloaders,
                    model=model,
                    processor=processor,
                    itm_head=itm_head,
                )
            callbacks.update({f"zeroshot_{k}":v for k,v in zeroshot_callbacks.items()})
        # if "linear_probe" in callbacks_config:
        if cfg.callbacks.get("linear_probe", None):
            linear_probe_callbacks = instantiate_linear_probe_callbacks(cfg.callbacks.linear_probe, dataloaders)
            callbacks.update({f"linear_{k}":v for k,v in linear_probe_callbacks.items()})
    
        # logger = WandbLogger(**logger_config)

        trainer = pl.Trainer(
            #devices=3,
            # logger=logger,
            callbacks=list(callbacks.values()),
            inference_mode=False,  # to allow the training of the linear probe
            **cfg.trainer,
        )

        trainer.validate(lit_model, datamodule=data_module)

        # dir_path = os.path.join("test_results", cfg.result_subdir)
        # os.makedirs(dir_path, exist_ok=True)
        # fpath = os.path.join(dir_path, f"{model_id}-{cfg.result_file_suffix}.p")

        # # Save zero-shot results as JSON (including config)
        # all_zeroshot_results = []
        # for key, value in data_dict.items():
        #     if key.startswith("zeroshot") and isinstance(value, list):
        #         all_zeroshot_results.extend(value)

        # if all_zeroshot_results:
        #     json_fpath = os.path.join(dir_path, f"{model_id}-zeroshot-all-{cfg.result_file_suffix}.json")
        #     json_dict = {
        #         "results": all_zeroshot_results,
        #         "config": OmegaConf.to_container(cfg, resolve=True)
        #     }
        #     with open(json_fpath, "w") as jf:
        #         json.dump(json_dict, jf, indent=2, default=str)  # default=str for numpy types

        # # Save the rest as pickle (excluding zero-shot results)
        # data_dict["__config__"] = cfg
        # pickle_dict = {k: v for k, v in data_dict.items() if not (k.startswith("zeroshot") and isinstance(v, list))}
        # if os.path.exists(fpath):
        #     with open(fpath, 'rb') as f:
        #         existing_data = pickle.load(f)
        #         for key, value in pickle_dict.items():
        #             if key not in existing_data:
        #                 existing_data[key] = value
        #         pickle_dict = existing_data
        # with open(fpath, 'wb') as f:
        #     pickle.dump(pickle_dict, f)


if __name__ == "__main__":
    main()
