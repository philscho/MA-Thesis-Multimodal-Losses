import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split, WeightedRandomSampler
import torchvision.datasets
from torchvision import transforms
from omegaconf import OmegaConf

from .datasets import (
    TorchVisionDatasetWrapper,
    Caltech101Dataset, 
    CocoDataset, 
    Cifar10Dataset, 
    ConceptualCaptionsDataset, 
    ImageNetADataset, 
    ImageNetDataset, 
    ImageNetValDataset,
    VisualGenomeDataset,
    Places365Dataset
)

class MyDataModule(L.LightningDataModule):
    def __init__(self, data_config, processor=None, augmentation=None, num_views=1, local_dev=False):
        super().__init__()
        self.config = data_config
        self.processor = processor
        self.augmentation = augmentation
        self.num_views = num_views
        self.test_datasets = None
        self.local_dev = local_dev

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self._setup_train_dataset()
            self.val_dataset = self._setup_val_dataset()
        elif stage == "validate":
            self.val_dataset = self._setup_val_dataset()
        else:
            raise NotImplementedError(f"Stage {stage} not implemented")

    #TODO: refactor
    def _setup_train_dataset(self):
        train_datasets = []
        if "coco" in self.config.dataset.train:
            coco = CocoDataset(
                root=self.config.dataset.coco.root,
                split="train",
                processor=self.processor if self.local_dev else None,
                transform=self.augmentation,
                num_views=self.num_views,
            )
            train_datasets.append(coco)
        if "cc3m" in self.config.dataset.train:
            cc3m = ConceptualCaptionsDataset(
                root=self.config.dataset.cc3m.data_dir,
                use_llava_split=False,
                processor=self.processor if self.local_dev else None,
                transform=self.augmentation,
                num_views=self.num_views,
            )
            train_datasets.append(cc3m)
        if "vg" in self.config.dataset.train:
            vg = VisualGenomeDataset(
                root=self.config.dataset.vg.data_dir,
                use_llava_split=True,
                processor=self.processor if self.local_dev else None,
                transform=self.augmentation,
                num_views=self.num_views,
            )
            train_datasets.append(vg)

        train_all = ConcatDataset(train_datasets)
        if self.config.dataset.use_subset.value:
            train_all = self._get_subset_dataset(train_all, self.config.dataset.use_subset.subset_fraction)
            print(f"Using a {self.config.dataset.use_subset.subset_fraction} subset of dataset")
        
        return train_all

    def _setup_val_dataset(self):
        if "coco_val" in self.config.dataset.val:
            return CocoDataset(
                root=self.config.dataset.coco.root,
                split="val",
                processor=self.processor if self.local_dev else None,
                transform=self.augmentation,
                num_views=self.num_views,
            )
        if "coco_val_dummy" in self.config.dataset.val:
            dataset = CocoDataset(
                root=self.config.dataset.coco.root,
                split="val",
                processor=self.processor if self.local_dev else None,
                transform=self.augmentation,
                num_views=self.num_views,
            )
            dataset = self._get_subset_dataset(dataset, 0.01)
            return dataset
        else:
            raise NotImplementedError(f"Val dataset {self.config.dataset.val} not implemented")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            **self.config.dataloader.train, 
            collate_fn=self._collate_fn if not self.local_dev else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            **self.config.dataloader.coco_val, 
            collate_fn=self._collate_fn if not self.local_dev else None,
        )
    
    def _setup_test_datasets(self):
        dataset_list = self.config.datasets
        dataset_dict = {dataset: dict(train=None, test=None) for dataset in dataset_list}
        for dataset_name in dataset_list:
            # Get dataset class object 
            dataset_kwargs = dict()
            if dataset_name == "Places365":
                dataset_class = Places365Dataset
                root = self.config[dataset_name]["root"]
            elif dataset_name == "ImageNet":
                # Use ImageNet validation set when testing on all 1000 classes
                dataset_class = ImageNetValDataset
            elif dataset_name in ["ImageNet-0.1", "ImageNet-0.01", "ImageNet-100", "ImageNet-100-0.1", "ImageNet-100-0.01"]:
                dataset_class = ImageNetDataset
            else:
                dataset_class = getattr(torchvision.datasets, dataset_name, None)
                dataset_kwargs["root"] = self.config.root
            
            if dataset_class is None:
                print(f"Dataset {dataset_name} not found.")
                dataset_dict.pop(dataset_name)
                continue
            
            if dataset_name in ["Caltech101", "Caltech256"]:
                dataset_object = dataset_class(**dataset_kwargs)
                train_size = int(0.8 * len(dataset_object))
                test_size = len(dataset_object) - train_size
                dataset_split = random_split(dataset_object, [train_size, test_size])
            
            if dataset_name in self.config:
                if "dataset_kwargs" in self.config[dataset_name]:
                    dataset_kwargs.update(self.config[dataset_name]["dataset_kwargs"])
            
            for split in ["train", "test"]:
                # Assign proper dataset-specific split kwarg
                if dataset_name in ["Places365"]:
                    if split == "train":
                        subdir = "train"
                        fraction=0.1
                    else:
                        subdir = "val"
                        fraction=1.0
                    dataset_kwargs["root"] = root + subdir
                    dataset_kwargs["fraction"] = fraction
                elif dataset_name in ["CIFAR10","CIFAR100"]:
                    dataset_kwargs["train"] = True if split == "train" else False
                elif dataset_name in ['OxfordIIITPet']:
                    dataset_kwargs['split'] = 'trainval' if split == 'train' else 'test'
                elif dataset_name in ["ImageNet"]:
                    pass # don't need to pass split kwarg
                elif dataset_name in ["ImageNet-0.1", "ImageNet-0.01", "ImageNet-100", "ImageNet-100-0.1", "ImageNet-100-0.01"]:
                    if split == "train":
                        if dataset_name == "ImageNet-0.1": dataset_kwargs["split"] = "10percent"
                        elif dataset_name == "ImageNet-0.01": dataset_kwargs["split"] = "1percent"
                        elif dataset_name == "ImageNet": dataset_kwargs["split"] = "train"
                        elif dataset_name == "ImageNet-100-0.1": dataset_kwargs["split"] = "100-10percent"
                        elif dataset_name == "ImageNet-100-0.01": dataset_kwargs["split"] = "100-1percent"
                    else:
                        if dataset_name == "ImageNet-100-0.1": dataset_kwargs["split"] = "100-val"
                        elif dataset_name == "ImageNet-100-0.01": dataset_kwargs["split"] = "100-val"
                        # else: kwargs["split"] = "100" # validate on 100 classes only
                else:
                    dataset_kwargs["split"] = split
                
                # Instantiate dataset object
                if dataset_name == "Caltech101" or dataset_name == "Caltech256":
                    dataset_object = dataset_split[0] if split == "train" else dataset_split[1]
                else:
                    dataset_object = dataset_class(**dataset_kwargs)
                
                if self.config.use_subset_probe.value:
                    print(f"Using a {self.config.use_subset_probe.subset_fraction} subset of dataset")
                    dataset_object = self._get_subset_dataset(dataset_object, self.config.use_subset_probe.subset_fraction)
                
                dataset_wrapper = TorchVisionDatasetWrapper(
                    dataset_object,
                    processor=self.processor,
                    label_as_caption=self.config.dataset.label_as_caption,
                    caption_template=self.config.dataset.caption_template,
                    )
                
                dataset_dict[dataset_name][f"{split}"] = dataset_wrapper
        
        self.test_datasets = dataset_dict
    
    def get_test_datasets(self):
        if self.test_datasets is None:
            self._setup_test_datasets()
        return self.test_datasets

    def get_test_dataloaders(self):
        datasets = self.get_test_datasets()
        dataloaders = {key: dict(train=None, test=None) for key in datasets.keys()}
        for name, splits in datasets.items():
            dataloaders[name]["train"] = DataLoader(splits["train"], **self.config.dataloader.test)
            dataloaders[name]["test"] = DataLoader(splits["test"], **self.config.dataloader.test)
        return dataloaders

    def get_class_balanced_sampler(dataset, fraction):
        # Calculate class weights
        class_counts = [0] * len(dataset.classes)
        for _, label in dataset.samples:
            class_counts[label] += 1
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [class_weights[label] for _, label in dataset.samples]
        num_samples = int(np.floor(len(dataset) * fraction))
        # Create the sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        return sampler
    

    # TODO: not sure if custom collate_fn works with pinned_memory in dataloader.
    # return type of processor is custom dict type, not sure if it works with pinned_memory
    def _collate_fn(self, batch):
        images, text = zip(*batch)
        images_1 = [image[0] for image in images]
        inputs = self.processor(
                    images=images_1, text=text, padding=True, truncation=True, return_tensors="pt"
        )
        if len(images[0]) == 2:
            images_2 = [image[1] for image in images]
            inputs_2 = self.processor(images=images_2, return_tensors="pt")
            inputs["pixel_values_2"] = inputs_2["pixel_values"]
        return inputs
        
        # flag  = isinstance(text,list) == True or isinstance(text,tuple) == True
        # assert flag==True, print (f'wrong. Text : {text}')
        # try:
        #     a = self.processor(
        #             images=images, text=text, padding=True, truncation=True, return_tensors="pt"
        #         )
        #     return a
        # except ValueError:
        #     print (text)
        #     print (type(text[0]))
        #     return 1
        # return a
    
    def _get_subset_dataset(self, dataset, fraction):
        num_samples = int(np.floor(len(dataset) * fraction))
        indices = np.random.default_rng(seed=42).choice(len(dataset), size=num_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices=indices)
        return subset
    
    def callback_dataloader(self):
        loaders = dict()
        if "cifar10" in self.config.dataset.val:
            dataset = Cifar10Dataset(
                    train=True,
                    processor=self.processor,
                    **self.config.dataset.cifar10,
            )
            if self.config.dataset.use_subset_probe.value:
                dataset = self._get_subset_dataset(dataset, self.config.dataset.use_subset_probe.subset_fraction)
                #print(f"Using a {self.config.dataset.use_subset_probe.subset_fraction} subset of dataset")
            loaders["cifar10_train"] = DataLoader(
                dataset=dataset,
                **self.config.dataloader.cifar10_val,
                # collate_fn=self._collate_fn if not self.local_dev else None,
            )
            dataset = Cifar10Dataset(
                    train=False,
                    processor=self.processor,
                    **self.config.dataset.cifar10,
            )
            if self.config.dataset.use_subset_probe.value:
                dataset = self._get_subset_dataset(dataset, self.config.dataset.use_subset_probe.subset_fraction)
                #print(f"Using a {self.config.dataset.use_subset_probe.subset_fraction} subset of dataset")
            loaders["cifar10_test"] = DataLoader(
                dataset=dataset,
                **self.config.dataloader.cifar10_val,
                # # collate_fn=self._collate_fn if not self.local_dev else None,
            )
        if "caltech101" in self.config.dataset.val:
            dataset = Caltech101Dataset(
                    processor=self.processor,
                    **self.config.dataset.caltech101,
            )
            if self.config.dataset.use_subset_probe.value:
                dataset = self._get_subset_dataset(dataset, self.config.dataset.use_subset_probe.subset_fraction)
                #print(f"Using a {self.config.dataset.use_subset_probe.subset_fraction} subset of dataset")
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            loaders["caltech101_train"] = DataLoader(
                dataset=train_dataset,
                **self.config.dataloader.caltech101_val,
                # collate_fn=self._collate_fn if not self.local_dev else None,
            )
            loaders["caltech101_test"] = DataLoader(
                dataset=test_dataset,
                **self.config.dataloader.caltech101_val,
                # # collate_fn=self._collate_fn if not self.local_dev else None,
            )
        if "imagenet" in self.config.dataset.val:
            dataset = ImageNetDataset(processor=self.processor)
            if self.config.dataset.use_subset_probe.value:
                dataset = self._get_subset_dataset(dataset, self.config.dataset.use_subset_probe.subset_fraction)
                #print(f"Using a {self.config.dataset.use_subset_probe.subset_fraction} subset of dataset")
            loaders["imagenet"] = DataLoader(
                dataset=dataset,
                **self.config.dataloader.imagenet,
                # collate_fn=self._collate_fn if not self.local_dev else None,
            )
        if "imagenet_a" in self.config.dataset.val:
            dataset = ImageNetADataset(processor=self.processor)
            if self.config.dataset.use_subset_probe.value:
                dataset = self._get_subset_dataset(dataset, self.config.dataset.use_subset_probe.subset_fraction)
                #print(f"Using a {self.config.dataset.use_subset_probe.subset_fraction} subset of dataset")
            loaders["imagenet_a"] = DataLoader(
                dataset=dataset,
                **self.config.dataloader.imagenet,
                # collate_fn=self._collate_fn if not self.local_dev else None,
            )
            loaders["imagenet-a_test"] = DataLoader(
                dataset=test_dataset,
                **self.config.dataloader.caltech101_val,
                # # collate_fn=self._collate_fn if not self.local_dev else None,
            )
        return loaders



if __name__ == "main":
    from omegaconf import OmegaConf
    import rootutils
    ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    data_config = OmegaConf.load("../../configs/data/test.yaml")

    dm = MyDataModule(data_config)
    test_dataloaders = dm.get_test_dataloaders()

