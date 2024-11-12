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
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    ViTConfig,
    BertConfig,
    VisionTextDualEncoderConfig,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor
)
from omegaconf import OmegaConf

from utils.callbacks import LinearProbeCallback, ZeroShotCallback
from utils.callbacks.utils import instantiate_linear_probe_callbacks, instantiate_zeroshot_callbacks
from data_module import MyDataModule
from model_module import LitMML
from utils.utils import EmptyDataset, LightningModelWrapper, log_callback_metrics

# --------------------------------- Setup ------------------------------------

config = OmegaConf.load('configs/config_g4_tests.yaml')

def get_models(config):
    model = VisionTextDualEncoderModel(
        config=VisionTextDualEncoderConfig.from_vision_text_configs(
            vision_config=ViTConfig(), text_config=BertConfig()
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

model, processor = get_models(config)
ckpt_path = config.checkpoint
lit_model = LitMML.load_from_checkpoint(ckpt_path, 
                                        model=model, 
                                        processor=processor,
                                        # loss_cfg=config.loss,
                                        # optimizer_cfg=config.optimizer,
                                        # scheduler_cfg=config.scheduler,
                                        # strict=False,
                                        )
model, processor = lit_model.model, lit_model.processor

data_module = MyDataModule(config, 
                           processor, 
                           augmentation=transforms.RandAugment(),
                           num_views=2,
                           )
# data_module.setup()
# val_dataloader = data_module.val_dataloader()
dataloaders = data_module.get_test_dataloaders()

callbacks = list()
zeroshot_callbacks = instantiate_zeroshot_callbacks(config.zeroshot, dataloaders, model, processor)
callbacks.extend(list(zeroshot_callbacks.values()))
linear_probe_callbacks = instantiate_linear_probe_callbacks(config.linear_probe, dataloaders)
callbacks.extend(list(linear_probe_callbacks.values()))

# --------------------------------- Training ---------------------------------

torch.set_float32_matmul_precision("medium")
seed_everything(config.lightning.seed, workers=True)

wandb_logger = WandbLogger(**config.wandb)
if rank_zero_only.rank == 0:
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    wandb_logger.experiment.config.update(cfg_dict)

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

#model_lightning = LightningModelWrapper(model)
#empty_dataloader = torch.utils.data.DataLoader(EmptyDataset())

trainer.validate(lit_model, datamodule=data_module)

# --------------------------------- Logging ----------------------------------

# Log zero-shot results
columns, data = log_callback_metrics(model_name=config.wandb.name,
                                     callbacks=zeroshot_callbacks.values(), 
                                     logger=wandb_logger, 
                                     config=config)
wandb_logger.log_table(key="zero-shot", columns=columns, data=data)

# Log linear probe results
columns, data = log_callback_metrics(model_name=config.wandb.name,
                                     callbacks=linear_probe_callbacks.values(), 
                                     logger=wandb_logger, 
                                     config=config)
wandb_logger.log_table(key="linear_probe", columns=columns, data=data)

