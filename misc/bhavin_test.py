#%%


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


def load_json(filepath):
    json_objects = []
    with open(filepath, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            json_objects.append(json_object)
    return json_objects


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



cfg_path = "train_config.yaml"
config = OmegaConf.load(cfg_path)


if __name__ == '__main__':
    model, processor = get_models(config)
    train_dataset, val_dataset = get_datasets(config, processor)
    train_dataloader, val_dataloader = get_dataloaders(config, train_dataset, val_dataset)
    # batch = next(iter(train_dataloader))

    for i,batch in enumerate(train_dataloader):
        break


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    outs = []
    images, tokens, token_type, mask = batch
    model.train()

    fig, axs = plt.subplots(2, 10, figsize=(40,8))

    for ind in range(100):
        optimizer.zero_grad()
        outputs = model(
                    input_ids=tokens,
                    attention_mask=mask,
                    pixel_values=images,
                    token_type_ids=token_type,
                    return_loss=True
                )
        print(outputs['loss'])
        outputs['loss'].backward()
        print(model.vision_model.encoder.layer[0].output.dense.weight.grad)
        optimizer.step()
        loss_history.append(outputs['loss'].detach().numpy())
        # outs.append(outputs['logits_per_image'])

        axs[0,ind].hist(model.vision_model.encoder.layer[0].output.dense.weight.grad.flatten())
        axs[0,ind].set_title(torch.linalg.norm(model.vision_model.encoder.layer[0].output.dense.weight.grad))
        axs[1,ind].hist(model.vision_model.encoder.layer[10].output.dense.weight.grad.flatten())
        axs[1,ind].set_title(torch.linalg.norm(model.vision_model.encoder.layer[10].output.dense.weight.grad))


    plt.plot(loss_history)
    plt.title('loss on one batch')
    plt.show()

#%%