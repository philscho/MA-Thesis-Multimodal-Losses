import os

import lightning as L
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms

from my_datasets import (
    Caltech101Dataset, CocoDataset, Cifar10Dataset, ConceptualCaptionsDataset, VisualGenomeDataset
)

class MyDataModule(L.LightningDataModule):
    def __init__(self, config, processor=None, augmentation=None, num_views=1, local_dev=False):
        super().__init__()
        self.config = config
        self.processor = processor
        self.augmentation = augmentation
        self.local_dev = local_dev
        self.num_views = num_views

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self._setup_train_dataset()
            self.val_dataset = self._setup_val_dataset()
        else:
            raise NotImplementedError(f"Stage {stage} not implemented")

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
    
    def callback_dataloader(self):
        loaders = {}
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
        return loaders

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
            
