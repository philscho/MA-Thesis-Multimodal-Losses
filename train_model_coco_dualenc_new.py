# %%

from typing import Tuple
import os

from omegaconf import OmegaConf
import numpy as np

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

from my_datasets import (
    Caltech101Dataset,
    CocoDataset,
    Cifar10Dataset,
    ConceptualCaptionsDataset,
    VisualGenomeDataset,
)
from utils.zero_shot_func import _create_zero_shot_classifier, _evaluate_zero_shot
from utils.utils import get_custom_cosine_schedule_with_warmup
from callbacks import LinearProbeCallback

from utils.optimizer_and_scheduler import get_optimizer, get_scheduler

from dotenv import load_dotenv

load_dotenv()


def get_datasets(config, processor=None):

    # image_transforms = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ColorJitter(brightness=.5, contrast=.2, saturation=.3, hue=.2),
    #         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
    # ])
    randaugment = transforms.RandAugment(**config.dataset.transforms.RandAugment)

    datasets = {}
    train_datasets, val_datasets = [], []

    if "coco" in config.dataset.train:
        coco = CocoDataset(
            config.dataset.coco.split_train,
            os.path.join(config.dataset.coco.data_dir, "train2014"),
            # processor=processor,
            processor=None,
            transform=randaugment,
        )
        train_datasets.append(coco)
        datasets["coco"] = coco
    if "cc3m" in config.dataset.train:
        cc3m = ConceptualCaptionsDataset(
            root=config.dataset.cc3m.data_dir,
            use_llava_split=False,
            processor=None,
            transform=randaugment,
        )
        train_datasets.append(cc3m)
        datasets["cc3m"] = cc3m
    if "vg" in config.dataset.train:
        vg = VisualGenomeDataset(
            root=config.dataset.vg.data_dir,
            use_llava_split=True,
            processor=None,
            transform=randaugment,
        )
        train_datasets.append(vg)
        datasets["vg"] = vg

    if "coco_val" in config.dataset.val:
        coco_val = CocoDataset(
            config.dataset.coco.split_val,
            os.path.join(config.dataset.coco.data_dir, "val2014"),
            processor=None,
        )
        val_datasets.append(coco_val)
        datasets["coco_val"] = coco_val
    if "cifar10_val" in config.dataset.val:
        cifar10_val = Cifar10Dataset(
            processor=processor,
            **config.dataset.cifar10,
        )
        val_datasets.append(cifar10_val)
        datasets["cifar10_val"] = cifar10_val
    if "caltech101_val" in config.dataset.val:
        caltech101_val = Caltech101Dataset(
            processor=processor,
            **config.dataset.caltech101,
        )
        val_datasets.append(caltech101_val)
        datasets["caltech101_val"] = caltech101_val

    train_all = ConcatDataset(train_datasets)
    if config.dataset.use_subset.value == True:
        num_samples = int(
            np.floor(len(train_all) * config.dataset.use_subset.subset_fraction)
        )
        indices = np.random.default_rng(seed=42).choice(
            len(train_all) + 1,
            size=num_samples,
            replace=False,
        )
        print(f"Using a {config.dataset.use_subset.subset_fraction} subset of dataset")
        train_all = torch.utils.data.Subset(train_all, indices=indices)

    datasets["train_all"] = train_all
    datasets["val_all"] = val_datasets
    return datasets


def collate_fn(batch, processor):
    images, text = zip(*batch)
    return processor(
        images=images, text=text, padding=True, truncation=True, return_tensors="pt"
    )


# def batch_input_processing_func(processor):
#     # Return a lambda that calls the top-level collate_fn with the processor
#     return lambda batch: collate_fn(batch, processor)


def batch_input_processing_func(processor):
    def collate_fn(batch):
        images, text = zip(*batch)
        return processor(
            images=images, text=text, padding=True, truncation=True, return_tensors="pt"
        )

    return collate_fn


def get_dataloaders(config, datasets, collate_fn=None):
    dataloaders = {}
    dataloaders["train_all"] = DataLoader(
        datasets["train_all"], collate_fn=collate_fn, **config.dataloader.train
    )
    dataloaders["coco_val"] = DataLoader(
        datasets["coco_val"], collate_fn=collate_fn, **config.dataloader.coco_val
    )
    dataloaders["cifar10_val"] = DataLoader(
        datasets["cifar10_val"], **config.dataloader.cifar10_val
    )
    dataloaders["caltech101_val"] = DataLoader(
        datasets["caltech101_val"], **config.dataloader.caltech101_val
    )
    return dataloaders


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
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.loss_cfg = loss_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.contrastive_loss = ContrastiveLossWithTemperature()
        self.model.logit_scale.requires_grad = False
        # self.matching_loss = nn.BCEWithLogitsLoss()
        # self.validation_step_outputs = []
        self.cifar_results = []
        # self.temperature = temperature
        # self.learning_rate = learning_rate
        self.cifar10_classifier = None
        if "image_text_matching" in self.loss_cfg.losses:
            self.matching_loss = nn.CrossEntropyLoss()
            self.itm_head = nn.Sequential(
                nn.Linear(
                    self.model.config.projection_dim * 2, 512
                ),  # TODO: make dims variable
                nn.ReLU(),
                nn.Linear(512, 2),
            )
        self.save_hyperparameters(
            ignore=["model", "image_encoder", "text_encoder", "tokenizer"]
        )

    def common_step(self, batch):
        outputs = self.model(**batch)       
        # Ouptut embeddings are already normalized
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds

        losses, metrics = {}, {}
        if "contrastive" in self.loss_cfg.losses:
            #loss = clip_contrastive_loss(image_out, text_out, self.loss_cfg.temperature).mean()
            #loss = outputs['loss']
            loss_contrastive = self.contrastive_loss(image_embeds, text_embeds)
            accuracy_contrastive = self.calculate_accuracy(image_embeds, text_embeds)
            losses["loss-contrastive"] = loss_contrastive
            metrics["acc-contrastive"] = accuracy_contrastive
            
        #TODO: put loss in seperate function
        if "image_text_matching" in self.loss_cfg.losses:
            bs = image_embeds.size(0)
            neg_image_embeds, neg_text_embeds = self._neg_embeddings(
                image_embeds,
                text_embeds,
                outputs.logits_per_image,
                outputs.logits_per_text,
            )
            selection = torch.randint(0, 2, (bs,)).to(image_embeds.device)
            selected_text_embeds = torch.where(
                selection.unsqueeze(1) == 0, text_embeds, neg_text_embeds
            )
            multimodal_embeds = torch.concat(
                (image_embeds, selected_text_embeds), dim=1
            )
            logits = self.itm_head(multimodal_embeds)
            # probs = F.softmax(logits, dim=1)
            loss_matching = self.matching_loss(logits, selection.long())
            preds = logits.argmax(dim=1)
            accuracy_matching = (preds == selection).sum() / len(selection)
            losses["loss-matching"] = loss_matching
            metrics["acc-matching"] = accuracy_matching

        return losses, metrics

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        losses, metrics = self.common_step(batch)
        loss = sum(losses.values()) / len(losses)
        self.log("loss-train", loss, sync_dist=True, prog_bar=True)
        for d in [losses, metrics]:
            for name, val in d.items():
                self.log(f"{name}-train", val, sync_dist=True)
        self.log(f"temperature", self.contrastive_loss.logit_scale)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            losses, metrics = self.common_step(batch)
            loss = sum(losses.values()) / len(losses)
            self.log("loss-val", loss, sync_dist=True, prog_bar=True)
            for d in [losses, metrics]:
                for name, val in d.items():
                    self.log(f"{name}-val", val, sync_dist=True)
            return loss
        else:
            return None

    def on_validation_epoch_start(self):
        self.cifar10_classifier = _create_zero_shot_classifier(
            forward_func=lambda x: self.model.get_text_features(input_ids=x),
            classnames=[
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
            templates="a photo of a {}",
            tokenizer=self.processor,
        )
        self.caltech101_classifier = _create_zero_shot_classifier(
            forward_func=lambda x: self.model.get_text_features(input_ids=x),
            classnames=self.trainer.val_dataloaders[2].dataset.categories,
            templates="a photo of a {}",
            tokenizer=self.processor,
        )

    def on_validation_epoch_end(self):
        cifar_10_result = _evaluate_zero_shot(
            forward_func=lambda x: self.model.get_image_features(pixel_values=x),
            classifier=self.cifar10_classifier,
            dataloader=self.trainer.val_dataloaders[1],
            confusion_matrix=True,
            top_k=(1,),
        )
        caltech_101_result = _evaluate_zero_shot(
            forward_func=lambda x: self.model.get_image_features(pixel_values=x),
            classifier=self.caltech101_classifier,
            dataloader=self.trainer.val_dataloaders[2],
            confusion_matrix=True,
            top_k=(1,),
        )

        # TODO: refactor this
        for k, v in cifar_10_result.items():
            if k == "ConfusionMatrix":
                self.logger.log_image(
                    key="cifar-10-confusionmatrix", images=[v], caption=["ConfMatrix"]
                )
            else:
                self.log("cifar10-accuracy", v, sync_dist=False)
        for k, v in caltech_101_result.items():
            if k == "ConfusionMatrix":
                self.logger.log_image(
                    key="caltech-101-confusionmatrix",
                    images=[v],
                    caption=["ConfMatrix"],
                )
            else:
                self.log("caltech101-accuracy", v, sync_dist=False)

    def calculate_accuracy(self, images, texts):
        # all_gpus: global_batch_size, embedding_dim), labels: (local_batch_size)
        (images_all_gpus, texts_all_gpus, labels) = _gather_embeddings_and_labels(
            images, texts
        )
        # shape: (local_batch_size, global_batch_size)
        logits_per_image = torch.matmul(images, texts_all_gpus.transpose(0, 1))
        logits_per_text = torch.matmul(texts, images_all_gpus.transpose(0, 1))
        acc_per_image = (logits_per_image.argmax(dim=-1) == labels).sum()
        acc_per_text = (logits_per_text.argmax(dim=-1) == labels).sum()
        accuracy = (acc_per_image + acc_per_text) / 2 / logits_per_image.size(0)
        return accuracy

    def _neg_embeddings(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
        similarity_i2t: Tensor,
        similarity_t2i: Tensor,
        text_atts: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            bs = image_embeds.size(0)
            weights_i2t = F.softmax(similarity_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(similarity_t2i[:, :bs], dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        image_embeds_neg, text_embeds_neg, text_atts_neg = [], [], []
        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_t2i[b], 1).item())
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_i2t[b], 1).item())
            text_embeds_neg.append(text_embeds[neg_idx])
            # text_atts_neg.append(text_atts[neg_idx])   #TODO: implement text attention mask?
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, text_embeds_neg  # , text_atts_neg

    def configure_optimizers(self):

        optimizer = get_optimizer(self.optimizer_cfg, params=self.parameters())

        if self.scheduler_cfg.name is None:
            print("No scheduler provided, using only optimizer")
            return optimizer

        elif (
            self.scheduler_cfg.name == "CosineWarmup"
        ):  # TODO: abstract it out in the schedulers file
            num_warmup_steps = self.scheduler_cfg.kwargs.pop("num_warmup_steps")
            num_training_steps = self.scheduler_cfg.kwargs.pop("num_training_steps")
            if num_warmup_steps == "epoch":
                num_warmup_steps = (
                    self.trainer.estimated_stepping_batches / self.trainer.max_epochs
                )
            if num_training_steps == "all":
                num_training_steps = (
                    self.trainer.estimated_stepping_batches - num_warmup_steps
                )
            lr_scheduler_config = {
                "scheduler": get_custom_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    **self.scheduler_cfg.kwargs,
                ),
                "interval": self.scheduler_cfg.interval,
            }

        else:
            if self.scheduler_cfg.name == "ReduceLROnPlateau":
                assert self.scheduler_cfg.monitor is not None

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
        cifar10_val_dataloader,
        caltech101_val_dataloader,
    ]

    net = LitMML(
        model,
        processor,
        loss_cfg=config.loss,
        optimizer_cfg=config.optimizer,
        scheduler_cfg=config.scheduler,
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
        val_dataloader_idx=1,
        linear_probe=torch.nn.Linear(512, 10),
        log_str_prefix="cifar10",
        **config.lightning.cifar_linear_probe_callback,
    )
    caltech101linear = LinearProbeCallback(
        val_dataloader_idx=2,
        linear_probe=torch.nn.Linear(512, 101),
        log_str_prefix="caltech101",
        **config.lightning.caltech101_linear_probe_callback,
    )

    callbacks = [
        ckpt_callback,
        lr_callback,
        early_stopping,
        cifar10linear,
        caltech101linear,
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
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloaders,
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
