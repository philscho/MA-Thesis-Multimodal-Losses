import json
import torch

from . import ZeroShotCallback, LinearProbeCallback
from ..data.datasets.utils import imagenet_classnames, imagenet_a_classnames

def instantiate_zeroshot_callbacks(zeroshot_config, dataloaders, model, processor, itm_head=None):
    callbacks = dict()
    for dataset_name, splits in dataloaders.items():
        if dataset_name == "Places365":
            split = "test"
        else: split = "train"
        callbacks[dataset_name] = ZeroShotCallback(
            dataset_name=dataset_name,
            dataloader=splits[split],
            classnames=splits[split].dataset.classnames,
            tokenizer=processor,
            text_forward=lambda x, y: model.get_text_features(input_ids=x, attention_mask=y),
            modality_forward=lambda x: model.get_image_features(pixel_values=x),
            itm_head=itm_head,
            **zeroshot_config,
        )
    return callbacks


def instantiate_zeroshot_callbacks_for_all_text_layers(zeroshot_config, dataloaders, model, processor, itm_head=None):
    callbacks = dict()
    for dataset_name, splits in dataloaders.items():
        if dataset_name == "Places365":
            split = "test"
        else: split = "train"
        for i in range(12):
            callbacks[f"{dataset_name}-text_layer_{i}"] = ZeroShotCallback(
                dataset_name=dataset_name,
                dataloader=splits[split],
                classnames=splits[split].dataset.classnames,
                tokenizer=processor,
                text_forward=lambda x, y, i=i: model.text_projection(
                    model.text_model.pooler(
                        model.text_model(
                            input_ids=x, attention_mask=y, output_hidden_states=True
                        ).hidden_states[i+1]
                    )
                ), # take [CLS] token, and skip embedding layer output
                modality_forward=lambda x: model.get_image_features(pixel_values=x),
                itm_head=itm_head,
                **zeroshot_config,
            )
    return callbacks

def text_forward(model, input_ids, attention_mask, layer_idx):
    hidden_states = model.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
    hidden_states = hidden_states[layer_idx+1] # first layer is embedding layer
    pooled_output = model.text_model.pooler(hidden_states)
    output = model.text_projection(pooled_output)
    return output

def instantiate_linear_probe_callbacks(linear_probe_config, dataloaders):
    callbacks = dict()
    for dataset_name, splits in dataloaders.items():
        num_classes = splits["train"].dataset.num_classes
        callbacks[dataset_name] = LinearProbeCallback(
            dataset_name=dataset_name,
            train_dataloader=splits["train"],
            test_dataloader=splits["test"],
            num_classes=num_classes,
            **linear_probe_config,
        )
    return callbacks


def _instantiate_zeroshot_callbacks(config, dataloaders, model, processor):
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

def _instantiate_linear_probe_callbacks(config, dataloaders):
    callbacks = dict()

    if 'cifar10' in config.datasets:
        callbacks['cifar10'] = LinearProbeCallback(
            train_dataloader=dataloaders["cifar-10_train"],
            test_dataloader=dataloaders["cifar-10_test"],
            linear_probe=torch.nn.Linear(512, 10),
            **config.cifar10,
        )
    if 'caltech101' in config.datasets:
        callbacks['caltech101'] = LinearProbeCallback(
            train_dataloader=dataloaders["caltech-101_train"],
            test_dataloader=dataloaders["caltech-101_test"],
            linear_probe=torch.nn.Linear(512, 101),
            **config.caltech101,
        )
    if 'imagenet' in config.datasets:
        callbacks['imagenet'] = LinearProbeCallback(
            train_dataloader=dataloaders["imagenet_train"],
            test_dataloader=dataloaders["imagenet_test"],
            linear_probe=torch.nn.Linear(512, 1000),
            **config.imagenet,
        )
    if 'imagenet-a' in config.datasets:
        callbacks['imagenet-a'] = LinearProbeCallback(
            train_dataloader=dataloaders["imagenet-a_train"],
            test_dataloader=dataloaders["imagenet-a_test"],
            linear_probe=torch.nn.Linear(512, 200),
            **config.imagnet_a,
        )
    
    return callbacks
