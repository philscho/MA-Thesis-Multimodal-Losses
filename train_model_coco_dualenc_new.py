# %%

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
from torchvision import transforms

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


def main(config):
    seed_everything(config.lightning.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(**config.wandb)
    # log the config on the master node
    if rank_zero_only.rank == 0:
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        wandb_logger.experiment.config.update(cfg_dict)
    # wandb_logger.watch(net)

    model, processor = get_models(config)
    randaugment = transforms.RandAugment(**config.dataset.transforms.RandAugment)
    net = LitMML(
        model,
        processor,
        loss_cfg=config.loss,
        optimizer_cfg=config.optimizer,
        scheduler_cfg=config.scheduler,
        augmentation=randaugment,
    )
    if config.gradient_checkpointing:
        net.model.vision_model.gradient_checkpointing_enable()
        net.model.text_model.gradient_checkpointing_enable()

    data_module = MyDataModule(config, processor, local_dev=False)
    callback_dataloaders = data_module.callback_dataloader()
    cifar10_train, cifar10_test = callback_dataloaders["cifar10_train"], callback_dataloaders["cifar10_test"]
    caltech101_train, caltech101_test = callback_dataloaders["caltech101_train"], callback_dataloaders["caltech101_test"]
    cifar10linear = LinearProbeCallback(
        train_dataloader=cifar10_train,
        test_dataloader=cifar10_test,
        linear_probe=torch.nn.Linear(512, 10),
        num_classes=10,
        log_str_prefix="cifar10",
        **config.lightning.cifar_linear_probe_callback,
    )
    caltech101linear = LinearProbeCallback(
        train_dataloader=caltech101_train,
        test_dataloader=caltech101_test,
        linear_probe=torch.nn.Linear(512, 101),
        num_classes=101,
        log_str_prefix="caltech101",
        **config.lightning.caltech101_linear_probe_callback,
    )
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

    checkpoint = config.get("resume_checkpoint", None)
    ckpt_callback = ModelCheckpoint(
        **config.lightning.model_checkpoint_callback,
        monitor="loss-val",  # "loss-val"
        dirpath=f"{config.save_dir}/ckpts/{wandb_logger.experiment.id}",
        filename="ckpt-{epoch:02d}-{loss-val:.3f}",
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(
        monitor="loss-val", patience=10, verbose=True
    )

    callbacks = [
        ckpt_callback,
        lr_callback,
        early_stopping,
        cifar10linear,
        caltech101linear,
        cifar10zeroshot,
        caltech101zeroshot,
    ]



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
        ckpt_path=checkpoint,
    )


if __name__ == "__main__":

    # set environment variable torch_distributed_debug to INFO
    # import os
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # from torch.utils.collect_env import get_pretty_env_info
    # if rank_zero_only.rank == 0:
    #     print(get_pretty_env_info())

    # cfg_path = "./configs/train_config.yaml"
    #cfg_path = "configs/config_local.yaml"
    cfg_path = "./configs/config_HLR.yaml"

    config = OmegaConf.load(cfg_path)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    OmegaConf.resolve(config)

    # config.wandb.offline = True

    # print the final values and go...
    if rank_zero_only.rank == 0:
        print("--" * 30)
        print("Config for the run : ")
        print("--" * 30)
        print(OmegaConf.to_yaml(config))
        print("--" * 30)

    main(config)
