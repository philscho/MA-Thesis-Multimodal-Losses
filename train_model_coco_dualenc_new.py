# %%

from typing import Tuple
import os

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
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
    contrastive_loss_with_temperature,
    _gather_embeddings_and_labels,
)
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
import numpy as np
from omegaconf import OmegaConf

from callbacks import LinearProbeCallback, ZeroShotCallback
from my_datasets import (
    Caltech101Dataset,
    CocoDataset,
    Cifar10Dataset,
    ConceptualCaptionsDataset,
    VisualGenomeDataset,
)
from data_module import MyDataModule
from utils.loss_functions import NTXentLoss
from utils.utils import (
    calculate_accuracy, get_custom_cosine_schedule_with_warmup, 
    get_negative_embeddings, print_memory_usage,
)
from utils.optimizer_and_scheduler import get_optimizer, get_scheduler

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


class LitMML(pl.LightningModule):
    def __init__(
        self,
        model: VisionTextDualEncoderModel,
        processor: VisionTextDualEncoderProcessor,
        loss_cfg,
        optimizer_cfg,
        scheduler_cfg,
        augmentation=None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.loss_cfg = loss_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.model.logit_scale.requires_grad = False
        # self.validation_step_outputs = []
        # self.temperature = temperature
        # self.learning_rate = learning_rate
        self.augmentation = augmentation
        self.save_hyperparameters(
            ignore=["model", "image_encoder", "text_encoder", "tokenizer"]
        )
        self._set_loss_functions(loss_cfg)

    def _set_loss_functions(self, loss_cfg):
        self.contrastive_loss = ContrastiveLossWithTemperature()

        if "image_text_matching" in self.loss_cfg.losses:
            self.matching_loss = nn.CrossEntropyLoss()
            self.itm_head = nn.Sequential(
                nn.Linear(
                    self.model.config.projection_dim * 2, 512
                ),  # TODO: make dims variable
                nn.ReLU(),
                nn.Linear(512, 2),
            )
        if "SimCLR" in self.loss_cfg.losses:
            self.simclr_loss = NTXentLoss()

    def common_step(self, batch):
        print_memory_usage("Beginning of step:")
        token = batch.input_ids
        images = batch.pixel_values
        token_type_ids = batch.token_type_ids
        attention_mask = batch.attention_mask
        
        print_memory_usage("After loading batch:")
        images_v1 = self.augmentation(images.to(torch.uint8))
        if "SimCLR" in self.loss_cfg.losses:
            images_v2 = self.augmentation(images.to(torch.uint8))

        print_memory_usage("After augmenting images:")
        outputs = self.model(
            pixel_values=images_v1, 
            input_ids=token, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )

        print_memory_usage("After model forward pass:")
        #outputs = self.model(**batch)
        # Ouptut embeddings are already normalized
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds

        losses, metrics = {}, {}
        if "contrastive" in self.loss_cfg.losses:
            #loss = clip_contrastive_loss(image_out, text_out, self.loss_cfg.temperature).mean()
            #loss = outputs['loss']
            loss_contrastive = self.contrastive_loss(image_embeds, text_embeds)
            accuracy_contrastive = calculate_accuracy(image_embeds, text_embeds)
            losses["loss-contrastive"] = loss_contrastive
            metrics["acc-contrastive"] = accuracy_contrastive
        
        print_memory_usage("After calculating contrastive loss:")
        #TODO: put loss in seperate function
        if "image_text_matching" in self.loss_cfg.losses:
            bs = image_embeds.size(0)
            neg_image_embeds, neg_text_embeds = get_negative_embeddings(
                image_embeds, text_embeds, outputs.logits_per_image, outputs.logits_per_text)
            selection = torch.randint(0, 2, (bs,)).to(image_embeds.device)
            selected_text_embeds = torch.where(selection.unsqueeze(1) == 0, text_embeds, neg_text_embeds)
            multimodal_embeds = torch.concat((image_embeds, selected_text_embeds), dim=1)
            logits = self.itm_head(multimodal_embeds)
            #probs = F.softmax(logits, dim=1)
            loss_matching = self.matching_loss(logits, selection.long())
            preds = logits.argmax(dim=1)
            accuracy_matching = (preds == selection).sum() / len(selection)
            losses['loss-matching'] = loss_matching
            metrics["acc-matching"] = accuracy_matching

        print_memory_usage("After calculating matching loss:")
        if "SimCLR" in self.loss_cfg.losses:
            image_embeds_v2 = self.model.get_image_features(pixel_values=images_v2)
            image_embeds_v2 = image_embeds_v2 / image_embeds_v2.norm(dim=-1, keepdim=True) # need to be normalized
            loss_simclr = self.simclr_loss(image_embeds, image_embeds_v2, pl_module=self)
            #accuracy_simclr = calculate_accuracy(z_1, z_2)
            losses["loss-simclr"] = loss_simclr
            #metrics["acc-simclr"] = accuracy_simclr
        
        print_memory_usage("After calculating SimCLR loss:")
        return losses, metrics

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        losses, metrics = self.common_step(batch)
        loss = sum(losses.values()) / len(losses)
        self.log("loss-train", loss, sync_dist=True, prog_bar=True)
        self.log(f"temperature", self.contrastive_loss.logit_scale)
        self._log_losses_and_metrics(losses, metrics)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        losses, metrics = self.common_step(batch)
        loss = sum(losses.values()) / len(losses)
        self.log("loss-val", loss, sync_dist=True, prog_bar=True)
        self._log_losses_and_metrics(losses, metrics)
        return loss
                
    def _log_losses_and_metrics(self, losses, metrics):
        for d in [losses, metrics]:
            for name, val in d.items():
                self.log(f"{name}-val", val, sync_dist=True)


    # def on_validation_epoch_start(self):
    #     self.cifar10_classifier = _create_zero_shot_classifier(
    #         forward_func=lambda x: self.model.get_text_features(input_ids=x),
    #         classnames=[
    #             "airplane",
    #             "automobile",
    #             "bird",
    #             "cat",
    #             "deer",
    #             "dog",
    #             "frog",
    #             "horse",
    #             "ship",
    #             "truck",
    #         ],
    #         templates="a photo of a {}",
    #         tokenizer=self.processor,
    #     )
    #     self.caltech101_classifier = _create_zero_shot_classifier(
    #         forward_func=lambda x: self.model.get_text_features(input_ids=x),
    #         classnames=self.trainer.val_dataloaders[2].dataset.categories,
    #         templates="a photo of a {}",
    #         tokenizer=self.processor,
    #     )

    # def on_validation_epoch_end(self):
    #     cifar_10_result = _evaluate_zero_shot(
    #         forward_func=lambda x: self.model.get_image_features(pixel_values=x),
    #         classifier=self.cifar10_classifier,
    #         dataloader=self.trainer.val_dataloaders[1],
    #         confusion_matrix=True,
    #         top_k=(1,),
    #     )
    #     caltech_101_result = _evaluate_zero_shot(
    #         forward_func=lambda x: self.model.get_image_features(pixel_values=x),
    #         classifier=self.caltech101_classifier,
    #         dataloader=self.trainer.val_dataloaders[2],
    #         confusion_matrix=True,
    #         top_k=(1,),
    #     )

    #     # TODO: refactor this
    #     for k, v in cifar_10_result.items():
    #         if k == "ConfusionMatrix":
    #             self.logger.log_image(
    #                 key="cifar-10-confusionmatrix", images=[v], caption=["ConfMatrix"]
    #             )
    #         else:
    #             self.log("cifar10-accuracy", v, sync_dist=False)
    #     for k, v in caltech_101_result.items():
    #         if k == "ConfusionMatrix":
    #             self.logger.log_image(
    #                 key="caltech-101-confusionmatrix",
    #                 images=[v],
    #                 caption=["ConfMatrix"],
    #             )
    #         else:
    #             self.log("caltech101-accuracy", v, sync_dist=False)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_cfg, params=self.parameters())
        return optimizer

        if self.scheduler_cfg.name is None:
            print("No scheduler provided, using only optimizer")
            return optimizer        
        else:
            if self.scheduler_cfg.name == "ReduceLROnPlateau":
                assert self.scheduler_cfg.monitor is not None
            if self.scheduler_cfg.name == "CosineWarmup":
                num_warmup_steps = self.scheduler_cfg.kwargs.pop("num_warmup_steps")
                num_training_steps = self.scheduler_cfg.kwargs.pop("num_training_steps")
                if num_warmup_steps == "epoch" and num_training_steps == "all":
                    num_warmup_steps = (
                        self.trainer.estimated_stepping_batches / self.trainer.max_epochs
                    )
                    num_training_steps = (
                        self.trainer.estimated_stepping_batches - num_warmup_steps
                    )
                self.scheduler_cfg.kwargs.update({
                    "num_warmup_steps": num_warmup_steps,
                    "num_training_steps": num_training_steps
                })

            monitor_metric = self.scheduler_cfg.pop("monitor")
            interval = self.scheduler_cfg.pop("interval")
            scheduler = get_scheduler(self.scheduler_cfg, optim=optimizer)
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": monitor_metric,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }


def main(config):
    seed_everything(config.lightning.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    model, processor = get_models(config)

    data_module = MyDataModule(config, processor)

    dataloaders = get_dataloaders(
        config,
        datasets=get_datasets(config, processor=processor),
        collate_fn=batch_input_processing_func(processor),
        # processor=processor,
    )
    train_dataloader = dataloaders["train_all"]
    coco_val_dataloader = dataloaders["coco_val"]
    cifar10_val_dataloader = dataloaders["cifar10_val"]
    caltech101_val_dataloader = dataloaders["caltech101_val"]
    val_dataloaders = [
        coco_val_dataloader,
    ]

    randaugment = transforms.RandAugment(**config.dataset.transforms.RandAugment)

    net = LitMML(
        model,
        processor,
        loss_cfg=config.loss,
        optimizer_cfg=config.optimizer,
        scheduler_cfg=config.scheduler,
        augmentation=randaugment,
    )

    checkpoint = config.get("resume_checkpoint", None)

    wandb_logger = WandbLogger(**config.wandb)
    # log the config on the master node
    if rank_zero_only.rank == 0:
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        wandb_logger.experiment.config.update(cfg_dict)

    ckpt_callback = ModelCheckpoint(
        **config.lightning.model_checkpoint_callback,
        monitor="loss-val/dataloader_idx_0",  # "loss-val"
        dirpath=f"{config.save_dir}/ckpts/{wandb_logger.experiment.id}",
        filename="ckpt-{epoch:02d}-{loss-val/dataloader_idx_0:.3f}",
    )

    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(
        monitor="loss-val/dataloader_idx_0", patience=10, verbose=True
    )

    cifar10linear = LinearProbeCallback(
        dataloader=cifar10_val_dataloader,
        linear_probe=torch.nn.Linear(512, 10),
        log_str_prefix="cifar10",
        **config.lightning.cifar_linear_probe_callback,
    )
    caltech101linear = LinearProbeCallback(
        dataloader=caltech101_val_dataloader,
        linear_probe=torch.nn.Linear(512, 101),
        log_str_prefix="caltech101",
        **config.lightning.caltech101_linear_probe_callback,
    )

    zs_config = config.zeroshot
    cifar10zeroshot = ZeroShotCallback(
        dataset_name="cifar10",
        dataloader=cifar10_val_dataloader,
        #classnames=cifar10_val_dataloader.dataset.classnames,
        classnames=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
        templates=zs_config.templates,
        tokenizer=processor,
        text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
        modality_forward=lambda x: model.get_image_features(pixel_values=x),
        batch_size=config.dataloader.cifar10_val.batch_size,
        device="cuda",
        top_k=(1,),
        confusion_matrix=True,
        #verbose=True
    )

    callbacks = [
        ckpt_callback,
        lr_callback,
        early_stopping,
        cifar10linear,
        caltech101linear,
        cifar10zeroshot,
    ]

    trainer = pl.Trainer(
        **config.lightning.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
        inference_mode=False,  # to allow the training of the linear probe
    )

    if config.gradient_checkpointing:
        net.model.vision_model.gradient_checkpointing_enable()
        net.model.text_model.gradient_checkpointing_enable()

    # wandb_logger.watch(net)

    trainer.fit(
        net,
        # train_dataloaders=train_dataloader,
        # val_dataloaders=val_dataloaders,
        datamodule=data_module,
        ckpt_path=checkpoint,
    )


if __name__ == "__main__":

    # from torch.utils.collect_env import get_pretty_env_info
    # if rank_zero_only.rank == 0:
    #     print(get_pretty_env_info())

    # cfg_path = "./configs/train_config.yaml"
    # cfg_path = "configs/config_local.yaml"
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
