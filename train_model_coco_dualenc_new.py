import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
#from src.data import MultiModalH5PyDataset
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature

from omegaconf import OmegaConf
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import rank_zero_only


from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    BertConfig,
    ViTConfig,
    VisionTextDualEncoderConfig
)

from datasets import load_dataset

os.environ["WANDB_API_KEY"] = "800844ad00f42b535fdcf9c737992c387867a397"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


def plot_helpers(x):
    return np.transpose(x, (1, 2, 0))

def load_json(filepath):
    json_objects = []
    with open(filepath, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            json_objects.append(json_object)
    return json_objects

#print("current working directory: ", os.getcwd())

#train_captions, val_captions = coco_captions["train"], coco_captions["validation"]

#train_image_dir = 'C:\\Users\\sadua\\OneDrive\\Dokumente\\Studium\\Masterarbeit\\Multimodal Learning\\data\\coco\\train2014'
#val_image_dir = 'C:\\Users\\sadua\\OneDrive\\Dokumente\\Studium\\Masterarbeit\\Multimodal Learning\\data\\coco\\val2014'

# Create dataset instance
# train_dataset = CustomDataset(train_split, train_image_dir, transform=image_transforms)
# val_dataset = CustomDataset(train_split, val_image_dir, transform=image_transforms)
# train_dataset_small = train_dataset[:50]
# val_dataset_small = train_dataset[50:75]

class CustomDataset(Dataset):
    def __init__(self, split_file, image_dir, processor, transform=None):
        self.split = load_json(split_file)
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, idx):
        sample = self.split[idx]
        img_filenames = sample['filename']
        img_filename = img_filenames[0] if isinstance(img_filenames, list) else img_filenames
        img_path = os.path.join(self.image_dir, img_filename)       
        # Load image
        image = Image.open(img_path).convert('RGB')
        # Load text
        i = 0 #TODO make random
        caption = sample['sentences'][i]

        if self.processor:
            inputs = self.processor(images=image, text=caption, padding='max_length', max_length=128, return_tensors="pt")
            img = torch.squeeze(inputs['pixel_values'])
            tokens = torch.squeeze(inputs['input_ids'])
            token_type = torch.squeeze(inputs['token_type_ids'])
            mask = torch.squeeze(inputs['attention_mask'])

        if self.transform:
            img = self.transform(img)

        return img, tokens, token_type, mask
    

def get_datasets(config, processor):

    image_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.5, contrast=.2, saturation=.3, hue=.2),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
        ]
    )
    
    train_dataset = CustomDataset(
        config.dataset.split_train, 
        os.path.join(config.dataset.data_dir, "train2014"), 
        processor,
        # image_transforms
    )

    val_dataset = CustomDataset(
        config.dataset.split_val, 
        os.path.join(config.dataset.data_dir, "val2014"), 
        processor,
        # image_transforms
    )

    return train_dataset, val_dataset


def get_dataloaders(config, train_dataset, val_dataset):

    train_dataloader = DataLoader(train_dataset, **config.dataloader.train_dataloader)
    val_dataloader = DataLoader(val_dataset, **config.dataloader.val_dataloader)

    return train_dataloader, val_dataloader


def get_models(config):
    config_vision = ViTConfig()
    config_text = BertConfig()
    config_model = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config=config_vision, text_config=config_text)
    
    model = VisionTextDualEncoderModel(config_model)

    image_processor = AutoImageProcessor.from_pretrained(config.model.image_encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder_name, use_fast=False)
    processor = VisionTextDualEncoderProcessor(image_processor=image_processor, tokenizer=tokenizer)

    return model, processor


class LitMML(pl.LightningModule):
    # def __init__(self, image_encoder, text_encoder,tokenizer,temperature,learning_rate):
    def __init__(
        self,
        model,
        processor,
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
        # self.temperature = temperature
        # self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=["model", "image_encoder", "text_encoder", "tokenizer"])
    
    def training_step(self, batch, batch_idx):
        images, tokens, token_type, mask = batch
        
        outputs = self.model(
            input_ids=tokens,
            attention_mask=mask,
            pixel_values=images,
            token_type_ids=token_type,
            return_loss=True
        )

        image_out = torch.nn.functional.normalize(outputs['image_embeds'], dim=-1)
        text_out = torch.nn.functional.normalize(outputs['text_embeds'], dim=-1)

        # loss, logits_per_image = outputs.loss, outputs.logits_per_image
        # self.log("loss/train", loss.mean(), sync_dist=True)
        # return loss.mean()

        if self.loss_cfg.name == "contrastive_like_clip":
            #loss = clip_contrastive_loss(image_out, text_out, self.loss_cfg.temperature).mean()
            #loss = self.contrastive_loss(image_out, text_out)
            loss = outputs['loss']
            self.log("loss-train", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, tokens, token_type, mask = batch
        
        outputs = self.model(
            input_ids=tokens,
            attention_mask=mask,
            pixel_values=images,
            token_type_ids=token_type,
            return_loss=True
        )

        image_out = torch.nn.functional.normalize(outputs['image_embeds'], dim=-1)
        text_out = torch.nn.functional.normalize(outputs['text_embeds'], dim=-1)

        # loss, logits_per_image = outputs.loss, outputs.logits_per_image
        # self.log("loss/train", loss.mean(), sync_dist=True)
        # return loss.mean()
    
        if self.loss_cfg.name == "contrastive_like_clip":
            #loss = clip_contrastive_loss(image_out, text_out, self.loss_cfg.temperature).mean()
            #loss = self.contrastive_loss(image_out, text_out)
            loss = outputs['loss']
            self.log("loss-val", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):

        if self.optimizer_cfg.name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg.kwargs)
        elif self.optimizer_cfg.name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_cfg.kwargs)
        elif self.optimizer_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_cfg.kwargs)
        elif self.optimizer_cfg.name == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), **self.optimizer_cfg.kwargs)
        else:
            raise ValueError(
                f"Wrong optimizer name. Provided {self.optimizer_cfg.name} which doesn't exist"
            )

        # check if scheduler is to be used
        if self.scheduler_cfg.name is None:
            print("No scheduler provided, using only optimizer")
            return optimizer

        elif self.scheduler_cfg.name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.scheduler_cfg.kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        elif self.scheduler_cfg.name == "CosineAnnealingLRWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLRWarmRestarts(
                optimizer, **self.scheduler_cfg.kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        elif self.scheduler_cfg.name == "ReduceROnPlateau":
            monitor_metric = self.scheduler_cfg.kwargs.pop("monitor")
            scheduler = torch.optim.lr_scheduler.ReduceROnPlateau(
                optimizer, **self.scheduler_cfg.kwargs
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": monitor_metric},
            }
        else:
            raise ValueError(
                f"Wrong scheduler name. Provided {self.scheduler_cfg.name} which doesn't exist"
            )


def clip_contrastive_loss(image_out, text_out, temperature):
    # credits : https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py
    # TODO make temperature learnable
    logits = (text_out @ image_out.T) / temperature
    images_similarity = image_out @ image_out.T
    texts_similarity = text_out @ text_out.T
    targets = torch.nn.functional.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    texts_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    images_loss = torch.nn.functional.cross_entropy(
        logits.T, targets.T, reduction="none"
    )
    loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)

    return loss

def contrastive_loss(image_features, text_features):
    image_batch = torch.nn.functional.normalize(image_features, dim=-1)
    text_batch = torch.nn.functional.normalize(text_features, dim=-1)

    logits = torch.dot(image_batch, text_batch.T)
    
    pass


def main(config):
    seed_everything(config.lightning.seed, workers=True)

    #dataset = load_dataset("yerevann/coco-karpathy")
    
    model, processor = get_models(config)
    train_dataset, val_dataset = get_datasets(config, processor)
    train_dataloader, val_dataloader = get_dataloaders(config, train_dataset, val_dataset)

    net = LitMML(
        model,
        processor,
        loss_cfg=config.loss,
        optimizer_cfg=config.optimizer,
        scheduler_cfg=config.scheduler,
    )

    print(net.parameters)

    wandb_logger = WandbLogger(**config.wandb)

    # log the config on the master node
    if rank_zero_only.rank == 0:
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        wandb_logger.experiment.config.update(cfg_dict)

    ckpt_callback = ModelCheckpoint(
        #every_n_epochs=2,
        monitor="loss-val",
        #dirpath=f"{config.save_dir}/ckpts/{wandb_logger.experiment.id}",
        filename="ckpt-{epoch:02d}-{val_loss:.3f}",
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        **config.lightning.trainer,
        logger=wandb_logger,
        callbacks=[ckpt_callback, lr_callback]
    )

    wandb_logger.watch(net)

    trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    # import sys
    # temperature = float(sys.argv[1])
    # learning_rate = float(sys.argv[2])

    # print (f'Starting run with temperature : {temperature} and learning_rate : {learning_rate}')
    # print("current working directory: ", os.getcwd())
    
    cfg_path = "./train_config.yaml"

    config = OmegaConf.load(cfg_path)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    OmegaConf.resolve(config)

    # print the final values and go...
    # print("--" * 30)
    # print("Config for the run : ")
    # print("--" * 30)
    # print(OmegaConf.to_yaml(config))
    # print("--" * 30)

    main(config)

