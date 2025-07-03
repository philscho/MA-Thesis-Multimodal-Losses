# %%

from pathlib import Path
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy
import os
import torch
from torchvision import transforms
from omegaconf import OmegaConf
import pickle

from torchvision.datasets import (
    SBU,
    Caltech101,
    Caltech256,
    CIFAR100,
    CIFAR10,
    Cityscapes,
    OxfordIIITPet,
    StanfordCars,
    STL10,
    Flowers102,
    Food101,
    FGVCAircraft,
    DTD,
)

# from callbacks.zeroshot_callback import _create_zero_shot_classifier,_evaluate_zero_shot
from eval_model import get_model_data
from src.callbacks import LinearProbeCallback, ZeroShotCallback
from src.callbacks.utils import instantiate_zeroshot_callbacks, instantiate_linear_probe_callbacks
from src.data.data_module import MyDataModule
from src.model.model_module import LitMML
from src.model.utils import get_model_and_processor
from src.utils.utils import EmptyDataset, LightningModelWrapper



if __name__ == '__main__':
    seed_everything(69, workers=True)

    model_config = OmegaConf.load('configs/model/model.yaml')
    data_config = OmegaConf.load('configs/data/data.yaml')
    zeroshot_config = OmegaConf.load('configs/callbacks/zeroshot.yaml')
    linear_probe_config = OmegaConf.load('configs/callbacks/linear_probe.yaml')
    trainer_config = OmegaConf.load('configs/lightning/trainer.yaml')
    model_list = OmegaConf.load('configs/model/trained_models.yaml')
    
    # config = OmegaConf.merge(config)
    #OmegaConf.resolve(config)
    
    data_dict = {}
    for model_id in model_list:
        model_name = model_list[model_id]['name']
        print ('--')
        print (model_name)
        print ('--')
        data_dict[model_name] = get_model_data(
            model_id,
            model_config,
            data_config,
            zeroshot_config,
            linear_probe_config,
            trainer_config
        )
        
    with open('linear_results.p','wb') as f:
        pickle.dump(data_dict,f)
 # needs some fixingc