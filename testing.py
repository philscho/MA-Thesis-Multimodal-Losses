#%%
import torch
import torchvision.transforms as transforms

# %%

import argparse
import sys
from dotenv import load_dotenv
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    LearningRateFinder,
    EarlyStopping,
)
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy
import torch
import torchvision.transforms as transforms

from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    BertConfig,
    ViTConfig,
    VisionTextDualEncoderConfig,
    optimization,
)
from omegaconf import OmegaConf

from utils.callbacks import LinearProbeCallback, ZeroShotCallback
from data_module import MyDataModule
from model_module import LitMML

def get_models(config):
    
    vision_config=ViTConfig()
    if '384' in config.model.image_encoder_name:
        print ('Changing the vision config to load the model...')
        vision_config.image_size = 384
    
    model = VisionTextDualEncoderModel(
        config=VisionTextDualEncoderConfig.from_vision_text_configs(
            vision_config=vision_config, text_config=BertConfig()
        )
    )
    model = MLMWrapper(model) 
    
    # model = VisionTextDualEncoderModel.from_vision_text_pretrained(config.model.image_encoder_name, config.model.text_encoder_name)
    processor = VisionTextDualEncoderProcessor(
        image_processor=AutoImageProcessor.from_pretrained(
            config.model.image_encoder_name, input_data_format="channels_last"
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            config.model.text_encoder_name, **config.model.tokenizer
        ),
    )
    return model, processor



T = transforms.Compose([transforms.RandAugment(2,9),
                        transforms.RandomResizedCrop,
                        transforms.RandomHorizontalFlip()])


#%%