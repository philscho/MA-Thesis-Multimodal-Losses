import torch

from callbacks import ZeroShotCallback
from callbacks.linear_probe_callback import LinearProbeCallback
from my_datasets.utils import imagenet_classnames, imagenet_a_classnames

def instantiate_zeroshot_callbacks(config, dataloaders, model, processor):
    callbacks = dict()
    if 'cifar10' in config.datasets:
        callbacks['cifar10'] = ZeroShotCallback(
            dataloader=dataloaders["cifar10_train"],
            tokenizer=processor,
            text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
            #text_forward=lambda x: model.get_text_features(input_ids=x),
            modality_forward=lambda x: model.get_image_features(pixel_values=x),
            **config.cifar10,
        )
    if 'caltech101' in config.datasets:
        callbacks['caltech101'] = ZeroShotCallback(
            dataloader=dataloaders["caltech101_train"],
            tokenizer=processor,
            text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
            modality_forward=lambda x: model.get_image_features(pixel_values=x),
            **config.caltech101,
        )
    if 'imagenet' in config.datasets:
        callbacks['imagenet'] = ZeroShotCallback(
            dataloader=dataloaders["imagenet"],
            tokenizer=processor,
            text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
            modality_forward=lambda x: model.get_image_features(pixel_values=x),
            **config.imagenet,
            classnames=imagenet_classnames(),
        )
    if 'imagenet_a' in config.datasets:
        callbacks['imagenet_a'] = ZeroShotCallback(
            dataloader=dataloaders["imagenet_a"],
            tokenizer=processor,
            text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
            modality_forward=lambda x: model.get_image_features(pixel_values=x),
            **config.imagenet_a,
            classnames=imagenet_a_classnames(),
        )
    
    return callbacks

def instantiate_linear_probe_callbacks(config, dataloaders):
    callbacks = dict()

    if 'cifar10' in config.datasets:
        callbacks['cifar10'] = LinearProbeCallback(
            train_dataloader=dataloaders["cifar-10_train"],
            test_dataloader=dataloaders["cifar-10_test"],
            linear_probe=torch.nn.Linear(512, 10),
            **config.cifar-10,
        )
    if 'caltech101' in config.datasets:
        callbacks['caltech101'] = LinearProbeCallback(
            train_dataloader=dataloaders["caltech-101_train"],
            test_dataloader=dataloaders["caltech-101_test"],
            linear_probe=torch.nn.Linear(512, 101),
            **config.caltech-101,
        )
    
    return callbacks
