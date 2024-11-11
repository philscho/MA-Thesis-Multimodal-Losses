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

from callbacks import LinearProbeCallback, ZeroShotCallback
from data_module import MyDataModule
from model_module import LitMML

load_dotenv()


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


def main(config):
    torch.set_float32_matmul_precision("medium")
    seed_everything(config.lightning.seed, workers=True)
    
    wandb_logger = WandbLogger(**config.wandb)
    if rank_zero_only.rank == 0:
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        wandb_logger.experiment.config.update(cfg_dict)

    model, processor = get_models(config)
    
    if config.dataset.transforms.enabled:
        augmentation = transforms.RandAugment(**config.dataset.transforms.RandAugment)
    else:
        augmentation = None
    
    net = LitMML(
        model,
        processor,
        loss_cfg=config.loss,
        optimizer_cfg=config.optimizer,
        scheduler_cfg=config.scheduler,
        #augmentation=augmentation,
    )
    # wandb_logger.watch(net)
    callbacks = []
    if config.gradient_checkpointing:
        net.model.vision_model.gradient_checkpointing_enable()
        net.model.text_model.gradient_checkpointing_enable()

    data_module = MyDataModule(config,
                               processor,
                               local_dev=False,
                               augmentation=augmentation,
                               num_views=2,
    )
    callback_dataloaders = data_module.callback_dataloader() 
    # callback_dataloaders = data_module.get_test_dataloaders()
    if 'caltech101' in config.dataset.val:
        caltech101_train = callback_dataloaders["caltech101_train"]
        caltech101_test = callback_dataloaders["caltech101_test"]
    if 'cifar10' in config.dataset.val:
        cifar10_train = callback_dataloaders["cifar10_train"]
        cifar10_test = callback_dataloaders["cifar10_test"]
        
    if 'cifar_linear_probe_callback' in config:
        cifar10linear = LinearProbeCallback(
            train_dataloader=cifar10_train,
            test_dataloader=cifar10_test,
            linear_probe=torch.nn.Linear(512, 10),
            **config.cifar_linear_probe_callback,
        )
        callbacks.append(cifar10linear)
    if 'caltech101_linear_probe_callback' in config:
        caltech101linear = LinearProbeCallback(
            train_dataloader=caltech101_train,
            test_dataloader=caltech101_test,
            linear_probe=torch.nn.Linear(512, 101),
            **config.caltech101_linear_probe_callback,
        )
        callbacks.append(caltech101linear)
    if 'cifar10_zeroshot_callback' in config:
        cifar10zeroshot = ZeroShotCallback(
            dataset_name="cifar10",
            dataloader=cifar10_train,
            #classnames=cifar10_val_dataloader.dataset.classnames,
            classnames=config.dataset.categories.cifar10,
            templates=config.zeroshot.templates,
            tokenizer=processor,
            text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
            modality_forward=lambda x: model.get_image_features(pixel_values=x),
            batch_size=config.dataloader.cifar10_val.batch_size,
            device="cuda",
            top_k=(1,),
            confusion_matrix=True,
            #verbose=True
        )
        callbacks.append(cifar10zeroshot)
    if 'caltech101_zeroshot_callback' in config:
        caltech101zeroshot = ZeroShotCallback(
            dataset_name="caltech101",
            dataloader=caltech101_train,
            #classnames=cifar10_val_dataloader.dataset.classnames,
            classnames=config.dataset.categories.caltech101,
            templates=config.zeroshot.templates,
            tokenizer=processor,
            text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
            modality_forward=lambda x: model.get_image_features(pixel_values=x),
            batch_size=config.dataloader.caltech101_val.batch_size,
            device="cuda",
            top_k=(1,),
            confusion_matrix=True,
            #verbose=True
        )
        callbacks.append(caltech101zeroshot)

    if "model_checkpoint_callback" in config:
        ckpt_callback = ModelCheckpoint(
            **config.model_checkpoint_callback,
            monitor="loss-val",  # "loss-val"
            dirpath=f"{config.save_dir}/ckpts/{wandb_logger.experiment.id}",
            filename="ckpt-{epoch:02d}-{loss-val:.3f}",
        )
        callbacks.append(ckpt_callback)
    
    lr_callback = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_callback) 
    
    if "early_stopping" in config:
        early_stopping = EarlyStopping(
            monitor="loss-val", 
            patience=10, 
            verbose=True,
        )
        callbacks.append(early_stopping)       

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        inference_mode=False,  # to allow the training of the linear probe
        strategy=DDPStrategy(static_graph=True)
            if config.lightning.trainer.pop("static_graph") and
                config.lightning.trainer.pop("strategy") == "ddp"
            else config.lightning.trainer.pop("strategy"),
        **config.lightning.trainer,
    )   
    
    trainer.fit(
        net,
        # train_dataloaders=train_dataloader,
        # val_dataloaders=val_dataloaders,
        datamodule=data_module,
        ckpt_path=config.get("resume_checkpoint", None),
    )





if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train a multimodal model.")
    # parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    # args = parser.parse_args()


    #---------OmegaConf Setup--------------
    # allows me to change parameters using cli without opening configs
    cfg_path = 'configs/hessian_train_config.yaml'
    print(f"Running with config: {cfg_path}")
    config = OmegaConf.load(cfg_path)
    categories_config = OmegaConf.load('configs/categories.yaml')
        
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    OmegaConf.resolve(config)
    #-------------------------------------
     
    # set environment variable torch_distributed_debug to INFO
    # import os
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # from torch.utils.collect_env import get_pretty_env_info
    # if rank_zero_only.rank == 0:
    #     print(get_pretty_env_info())

    # print the final values and go...
    if rank_zero_only.rank == 0:
        print("--" * 30)
        print("Config for the run : ")
        print("--" * 30)
        print(OmegaConf.to_yaml(config))
        print("--" * 30)
        
    config = OmegaConf.merge(config,categories_config)

    
    main(config)
