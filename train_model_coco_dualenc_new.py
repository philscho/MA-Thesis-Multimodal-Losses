#%%

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
    ContrastiveLossWithTemperature, contrastive_loss_with_temperature, _gather_embeddings_and_labels
)

import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, LearningRateFinder, EarlyStopping
from lightning.pytorch.utilities import rank_zero_only

from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    BertConfig,
    ViTConfig,
    VisionTextDualEncoderConfig,
    optimization
)

from my_datasets import CocoDataset, Cifar10Dataset, ConceptualCaptionsDataset, VisualGenomeDataset
from utils.zero_shot_func import _create_zero_shot_classifier, _evaluate_zero_shot
from utils.utils import get_custom_cosine_schedule_with_warmup
from .callbacks import CIFAR10LinearProbeCallback

os.environ["WANDB_API_KEY"] = ""
#os.environ["TOKENIZERS_PARALLELISM"] = "false"



def get_datasets(config, processor=None):

    image_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.5, contrast=.2, saturation=.3, hue=.2),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
    ])

    randaugment = transforms.RandAugment(**config.dataset.transforms.RandAugment)

    coco = CocoDataset(
        config.dataset.coco.split_train, 
        os.path.join(config.dataset.coco.data_dir, "train2014"), 
        #processor=processor,
        processor=None,
        transform=randaugment
    )
    cc3m = ConceptualCaptionsDataset(root=config.dataset.cc3m.data_dir, use_llava_split=True, 
                                     processor=None, transform=randaugment)
    vg = VisualGenomeDataset(root=config.dataset.vg.data_dir, use_llava_split=True,
                             processor=None, transform=randaugment)
    
    coco_val = CocoDataset(
        config.dataset.coco.split_val, 
        os.path.join(config.dataset.coco.data_dir, "val2014"), 
        processor=None,
        # image_transforms
    )
    cifar10_val = Cifar10Dataset(processor=processor)
    #cifar10_val = Cifar10Dataset(transform=transforms.ToTensor())

    train_all = ConcatDataset([coco, cc3m, vg])

    datasets = {}
    datasets['train_all'] = train_all
    datasets['coco'] = coco
    datasets['cc3m'] = cc3m
    datasets['vg'] = vg
    datasets['coco_val'] = coco_val
    datasets['cifar10_val'] = cifar10_val
    return datasets


def get_dataloaders(config, datasets, collate_fn=None):
    dataloaders = {}
    dataloaders['train_all'] = DataLoader(datasets['train_all'], collate_fn=collate_fn, **config.dataloader.train)
    #dataloaders['train_all'] = DataLoader(datasets['vg'], **config.dataloader.train)
    # dataloaders['coco'] = DataLoader(datasets['coco'], **config.dataloader.train)
    # dataloaders['cc3m'] = DataLoader(datasets['cc3m'], **config.dataloader.train)
    # dataloaders['vg'] = DataLoader(datasets['vg'], **config.dataloader.train)
    dataloaders['coco_val']= DataLoader(datasets['coco_val'], collate_fn=collate_fn, **config.dataloader.coco_val)
    dataloaders['cifar10_val'] = DataLoader(datasets['cifar10_val'], **config.dataloader.cifar10_val)
    return dataloaders


def get_models(config):
    model = VisionTextDualEncoderModel(
        config=VisionTextDualEncoderConfig.from_vision_text_configs(
        vision_config=ViTConfig(), 
        text_config=BertConfig()
    ))

    # getting pretrained 
    # model = VisionTextDualEncoderModel.from_vision_text_pretrained(config.model.image_encoder_name, config.model.text_encoder_name)
    
    processor = VisionTextDualEncoderProcessor(
        image_processor=AutoImageProcessor.from_pretrained(config.model.image_encoder_name), 
        tokenizer=AutoTokenizer.from_pretrained(config.model.text_encoder_name, **config.model.tokenizer)
    )
    return model, processor


def batch_input_processing_func(processor):
    def collate_fn(batch):
            images, text = zip(*batch)
            data = processor(images=images, text=text, padding=True, truncation=True, return_tensors="pt")
            return data
    return collate_fn


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
        #self.matching_loss = nn.BCEWithLogitsLoss()  
        #self.validation_step_outputs = []
        self.cifar_results = []
        # self.temperature = temperature
        # self.learning_rate = learning_rate
        self.cifar10_classifier = None
        if "image_text_matching" in self.loss_cfg.losses:
            self.matching_loss = nn.CrossEntropyLoss()
            self.itm_head = nn.Sequential(
                nn.Linear(self.model.config.projection_dim * 2, 512),   #TODO: make dims variable
                nn.ReLU(),
                nn.Linear(512, 2)
            )
        self.save_hyperparameters(ignore=["model", "image_encoder", "text_encoder", "tokenizer"])
    
    def common_step(self, batch):

        outputs = self.model(**batch)
        
        # Ouptut embeddings are already normalized
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds
        # image_norm = torch.nn.functional.normalize(outputs['image_embeds'], dim=-1)
        # text_norm = torch.nn.functional.normalize(outputs['text_embeds'], dim=-1)

        losses, metrics = {}, {}
        #TODO: make loss function selection dynamic
        if "contrastive" in self.loss_cfg.losses:
            #loss = clip_contrastive_loss(image_out, text_out, self.loss_cfg.temperature).mean()
            loss_contrastive = self.contrastive_loss(image_embeds, text_embeds)
            #loss = outputs['loss']
            accuracy_contrastive = self.calculate_accuracy(image_embeds, text_embeds)
            losses["loss-contrastive"] = loss_contrastive
            metrics["acc-contrastive"] = accuracy_contrastive
            
        #TODO: put loss in seperate function
        #TODO: add accuracy for image text matching
        if "image_text_matching" in self.loss_cfg.losses:
            bs = image_embeds.size(0)
            neg_image_embeds, neg_text_embeds = self._neg_embeddings(
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

        # loss, logits_per_image = outputs.loss, outputs.logits_per_image
        # self.log("loss/train", loss.mean(), sync_dist=True)
        # return loss.mean()

        # loss, image_out, text_out = self.common_step(batch, batch_idx, dataloader_idx)
        # images, tokens, token_type, mask = batch
        
        # outputs = self.model(
        #     input_ids=tokens,
        #     attention_mask=mask,
        #     pixel_values=images,
        #     token_type_ids=token_type,
        #     #return_loss=True
        # )
        outputs = self.model(**batch)
        # Ouptut embeddings are already normalized
        image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds

        # image_norm = torch.nn.functional.normalize(outputs['image_embeds'], dim=-1)
        # text_norm = torch.nn.functional.normalize(outputs['text_embeds'], dim=-1)

        losses_all = []
        #TODO: make loss function selection dynamic
        if "contrastive" in self.loss_cfg.losses:
            #loss = clip_contrastive_loss(image_out, text_out, self.loss_cfg.temperature).mean()
            loss_contrastive = self.contrastive_loss(image_embeds, text_embeds)
            #loss = outputs['loss']
            accuracy = self.calculate_accuracy(image_embeds, text_embeds)
            self.log(f"loss-train-contrastive", loss_contrastive, sync_dist=True)
            self.log(f"acc-train", accuracy, sync_dist=True)
            self.log(f"temperature", self.contrastive_loss.logit_scale)

        #TODO: put loss in seperate function
        if "image_text_matching" in self.loss_cfg.losses:
            bs = image_embeds.size(0)
            neg_image_embeds, neg_text_embeds = self._neg_embeddings(
                image_embeds, text_embeds, outputs.logits_per_image, outputs.logits_per_text)
            selection = torch.randint(0, 2, (bs,)).to(image_embeds.device)
            selected_text_embeds = torch.where(selection.unsqueeze(1) == 0, text_embeds, neg_text_embeds)
            multimodal_embeds = torch.concat((image_embeds, selected_text_embeds), dim=1)

            logits = self.itm_head(multimodal_embeds)
            probs = F.softmax(logits, dim=1)
            loss_matching = self.matching_loss(probs, selection.long())
            self.log(f"loss-train-matching", loss_matching, sync_dist=True)
            losses_all.append(loss_matching)
        
        
        
        loss = sum(losses_all) / len(losses_all)
        self.log(f"loss-train", loss, sync_dist=True, prog_bar=True)
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
        
        if dataloader_idx == 0:
            # loss, image_out, text_out = self.common_step(batch, batch_idx)

            # loss, logits_per_image = outputs.loss, outputs.logits_per_image
            # self.log("loss/train", loss.mean(), sync_dist=True)
            # return loss.mean()

            # images, tokens, token_type, mask = batch
        
            # outputs = self.model(
            #     input_ids=tokens,
            #     attention_mask=mask,
            #     pixel_values=images,
            #     token_type_ids=token_type,
            #     #return_loss=True
            # )

            outputs = self.model(**batch)
            image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds

            # image_norm = torch.nn.functional.normalize(outputs['image_embeds'], dim=-1)
            # text_norm = torch.nn.functional.normalize(outputs['text_embeds'], dim=-1)
        
            losses_all = []
            if "contrastive" in self.loss_cfg.losses:
                loss_contrastive = self.contrastive_loss(image_embeds, text_embeds)
                accuracy = self.calculate_accuracy(image_embeds, text_embeds)
                self.log("loss-contrastive-val", loss_contrastive, sync_dist=True)
                self.log("acc-val", accuracy, sync_dist=True)
                losses_all.append(loss_contrastive)
            
            if "image_text_matching" in self.loss_cfg.losses:
                bs = image_embeds.size(0)
                neg_image_embeds, neg_text_embeds = self._neg_embeddings(
                    image_embeds, text_embeds, outputs.logits_per_image, outputs.logits_per_text)
                selection = torch.randint(0, 2, (bs,)).to(image_embeds.device)
                selected_text_embeds = torch.where(selection.unsqueeze(1) == 0, text_embeds, neg_text_embeds)
                multimodal_embeds = torch.concat((image_embeds, selected_text_embeds), dim=1)

                logits = self.itm_head(multimodal_embeds)
                probs = F.softmax(logits, dim=1)
                loss_matching = self.matching_loss(probs, selection.long())
                self.log(f"loss-val-matching", loss_matching, sync_dist=True)
                losses_all.append(loss_matching)
            
            loss = sum(losses_all) / len(losses_all)
            self.log(f"loss-val", loss, sync_dist=True, prog_bar=True)
            return loss

        # CIFAR-10 dataloader -- OG
        # if dataloader_idx == 1:
        #     images, tokens, token_type, mask, label = batch
        #     # Remove batch dimension from text embeddings (if batch_size = 1)
        #     tokens.squeeze_(0), token_type.squeeze_(0), mask.squeeze_(0)
            
        #     outputs = self.model(
        #         pixel_values=images,
        #         input_ids=tokens,
        #         attention_mask=mask,
        #         token_type_ids=token_type
        #     )

        #     image_out = torch.nn.functional.normalize(outputs['image_embeds'], dim=-1)
        #     text_out = torch.nn.functional.normalize(outputs['text_embeds'], dim=-1)
        #     # (b, 1, h_dim) x (b, h_dim, classes) => (b, 1, classes)
        #     #sim_scores = torch.squeeze(torch.bmm(image_out.unsqueeze_(1), text_out.transpose(1, 2)), 1)
        #     sim_scores = image_out @ text_out.transpose(0, 1)
        #     result = sim_scores.argmax(-1) == label
        #     #self.cifar_results.append(result.item())
        #     self.cifar_results.append(result)
        
        # if dataloader_idx == 1:
        #     images, label = batch
            
        #     image_features = self.model.get_image_features(pixel_values=images)
        #     image_out = torch.nn.functional.normalize(image_features, dim=-1)
            
        #     # (b, 1, h_dim) x (b, h_dim, classes) => (b, 1, classes)
        #     #sim_scores = torch.squeeze(torch.bmm(image_out.unsqueeze_(1), text_out.transpose(1, 2)), 1)
        #     sim_scores = image_out @ self.cifar10_classifier
        #     result = sim_scores.argmax(-1) == label
        #     #self.cifar_results.append(result.item())
        #     self.cifar_results.append(result)

    
    def on_validation_epoch_start(self):
        self.cifar10_classifier = _create_zero_shot_classifier(
            # forward_func=self.model.get_text_features,
            forward_func=lambda x: self.model.get_text_features(input_ids=x),
            classnames=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
            templates="a photo of a {}",
            tokenizer=self.processor
        )

    
    def on_validation_epoch_end(self):
        # #return 
        # preds = torch.stack(self.cifar_results).flatten()
        # #accuracy = sum(self.cifar_results) / len(self.cifar_results)
        # accuracy = preds.sum() / len(preds)
        # self.log("cifar10-accuracy", accuracy, sync_dist=True)
        # self.cifar_results = []

        result = _evaluate_zero_shot(
            forward_func=lambda x: self.model.get_image_features(pixel_values=x),
            classifier=self.cifar10_classifier,
            dataloader=self.trainer.val_dataloaders[1],
            confusion_matrix=True,
            top_k=(1,)
        )
        # assert len(result) == 1
        for k, v in result.items():
            if k=='ConfusionMatrix':
                self.logger.log_image(key='cifar-10-confusionmatrix',images=[v],caption=['ConfMatrix'])
            else:
                self.log("cifar10-accuracy", v, sync_dist=False)

    
    def calculate_accuracy(self, images, texts):
        # all_gpus: global_batch_size, embedding_dim), labels: (local_batch_size)
        (images_all_gpus, texts_all_gpus, labels) = _gather_embeddings_and_labels(images, texts)
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
        text_atts: Tensor = None
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
            #text_atts_neg.append(text_atts[neg_idx])   #TODO: implement text attention mask?
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        #text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, text_embeds_neg #, text_atts_neg
    
    def configure_optimizers(self):
        if self.optimizer_cfg.name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr, **self.optimizer_cfg.kwargs)
        elif self.optimizer_cfg.name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizer_cfg.lr, **self.optimizer_cfg.kwargs)
        elif self.optimizer_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_cfg.lr, **self.optimizer_cfg.kwargs)
        elif self.optimizer_cfg.name == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.optimizer_cfg.lr, **self.optimizer_cfg.kwargs)
        else:
            raise ValueError(f"Wrong optimizer name. Provided {self.optimizer_cfg.name} which doesn't exist")

        schedulers = []
        if self.scheduler_cfg.name is None:
            print("No scheduler provided, using only optimizer")
            return optimizer
        if "CosineAnnealingLR" in self.scheduler_cfg.name:
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_cfg.kwargs))
        if "CosineAnnealingLRWarmRestarts" in self.scheduler_cfg.name:
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.scheduler_cfg.kwargs))
        if "CosineWarmup" in self.scheduler_cfg.name:
            num_warmup_steps = self.scheduler_cfg.kwargs.pop("num_warmup_steps")
            num_training_steps = self.scheduler_cfg.kwargs.pop("num_training_steps")
            if num_warmup_steps == "epoch":
                num_warmup_steps = self.trainer.estimated_stepping_batches / self.trainer.max_epochs
            if num_training_steps == "all":
                num_training_steps = self.trainer.estimated_stepping_batches - num_warmup_steps
            lr_scheduler_config = {
                "scheduler": get_custom_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **self.scheduler_cfg.kwargs),
                "interval": self.scheduler_cfg.interval
            }
            schedulers.append(lr_scheduler_config)
        if "CosineWarmupHardRestarts" in self.scheduler_cfg.name:
            if self.scheduler_cfg.kwargs.num_warmup_steps == "epoch":
                num_warmup_steps = self.trainer.estimated_stepping_batches / self.trainer.max_epochs
                del self.scheduler_cfg.kwargs["num_warmup_steps"]
            if self.scheduler_cfg.kwargs.num_training_steps == "all":
                num_training_steps = self.trainer.estimated_stepping_batches - num_warmup_steps
                del self.scheduler_cfg.kwargs["num_training_steps"]
            schedulers.append(optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **self.scheduler_cfg.kwargs))
        if "ReduceLROnPlateau" in self.scheduler_cfg.name:
            monitor_metric = self.scheduler_cfg.kwargs.pop("monitor")
            schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_cfg.kwargs))
            return {"optimizer": optimizer, 
                    "lr_scheduler": {"scheduler": schedulers, "monitor": monitor_metric, "interval": self.scheduler_cfg.interval}}
        # else:
        #     raise ValueError(f"Wrong scheduler name. Provided {self.scheduler_cfg.name} which doesn't exist")
        
        # return {"optimizer": optimizer, 
        #         "lr_scheduler": {"scheduler": schedulers, "interval": self.scheduler_cfg.interval}}
        return [optimizer], schedulers


def main(config):
    seed_everything(config.lightning.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    model, processor = get_models(config)
   
    #dataloaders = get_dataloaders(config, get_datasets(config, processor))
    dataloaders = get_dataloaders(config, datasets=get_datasets(config, processor=processor), collate_fn=batch_input_processing_func(processor))
    train_dataloader = dataloaders['train_all']
    coco_val_dataloader = dataloaders['coco_val']
    cifar10_val_dataloader = dataloaders['cifar10_val']
    val_dataloaders = [coco_val_dataloader, cifar10_val_dataloader]
    #val_dataloaders = coco_val_dataloader

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
        monitor="loss-val/dataloader_idx_0", #"loss-val"
        dirpath=f"{config.save_dir}/ckpts/{wandb_logger.experiment.id}",
        filename="ckpt-{epoch:02d}-{loss-val/dataloader_idx_0:.3f}",
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="loss-val/dataloader_idx_0", patience=10,verbose=True)
    
    cifar10linear = CIFAR10LinearProbeCallback(**config.lightning.cifar_linear_probe_callback)
    callbacks=[ckpt_callback, lr_callback, early_stopping, cifar10linear] #[ckpt_callback, lr_callback]


    trainer = pl.Trainer(
        **config.lightning.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
        inference_mode=False, #to allow the training of the linear probe
    )

    if config.gradient_checkpointing:
        net.model.vision_model.gradient_checkpointing_enable() 
        net.model.text_model.gradient_checkpointing_enable()

    #wandb_logger.watch(net)

    trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders, ckpt_path=checkpoint)


if __name__ == "__main__":
    
    cfg_path = "configs/train_config.yaml"

    config = OmegaConf.load(cfg_path)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    OmegaConf.resolve(config)

    # config.wandb.offline = True

    # print the final values and go...
    print("--" * 30)
    print("Config for the run : ")
    print("--" * 30)
    print(OmegaConf.to_yaml(config))
    print("--" * 30)

    main(config)

