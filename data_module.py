import os

import lightning as L
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from my_datasets import (
    Caltech101Dataset, CocoDataset, Cifar10Dataset, ConceptualCaptionsDataset, VisualGenomeDataset
)

class MyDataModule(L.LightningDataModule):
    def __init__(self, config, processor=None):
        super().__init__()
        self.config = config
        self.processor = processor
        self.randaugment = transforms.RandAugment(**config.dataset.transforms.RandAugment)

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
                self.config.dataset.coco.split_train,
                os.path.join(self.config.dataset.coco.data_dir, "train2014"),
                processor=None,
            )
            train_datasets.append(coco)
        if "cc3m" in self.config.dataset.train:
            cc3m = ConceptualCaptionsDataset(
                root=self.config.dataset.cc3m.data_dir,
                use_llava_split=False,
                processor=None,
            )
            train_datasets.append(cc3m)
        if "vg" in self.config.dataset.train:
            vg = VisualGenomeDataset(
                root=self.config.dataset.vg.data_dir,
                use_llava_split=True,
                processor=None,
            )
            train_datasets.append(vg)

        train_all = ConcatDataset(train_datasets)
        if self.config.dataset.use_subset.value:
            num_samples = int(np.floor(len(train_all) * self.config.dataset.use_subset.subset_fraction))
            indices = np.random.default_rng(seed=42).choice(len(train_all), size=num_samples, replace=False)
            train_all = torch.utils.data.Subset(train_all, indices=indices)
            print(f"Using a {self.config.dataset.use_subset.subset_fraction} subset of dataset")

        return train_all

    def _setup_val_dataset(self):
        if "coco_val" in self.config.dataset.val:
            return CocoDataset(
                self.config.dataset.coco.split_val,
                os.path.join(self.config.dataset.coco.data_dir, "val2014"),
                processor=None,
            )
        else:
            raise NotImplementedError(f"Val dataset {self.config.dataset.val} not implemented")

    def _setup_callback_datasets(self):
        datasets = {}
        if "cifar10_val" in self.config.dataset.val:
            datasets["cifar10_val"] = Cifar10Dataset(
                processor=self.processor,
                **self.config.dataset.cifar10,
            )
        if "caltech101_val" in self.config.dataset.val:
            datasets["caltech101_val"] = Caltech101Dataset(
                processor=self.processor,
                **self.config.dataset.caltech101,
            )
        return datasets

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.config.dataloader.train, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.config.dataloader.coco_val, collate_fn=self._collate_fn)
    
    def callback_dataloader(self):
        loaders = {}
        if "cifar10_val" in self.config.dataset.val:
            loaders["cifar10_val"] = DataLoader(
                Cifar10Dataset(
                    processor=self.processor,
                    **self.config.dataset.cifar10,
                ),
                **self.config.dataloader.cifar10_val,
                collate_fn=self._collate_fn,
            )
        if "caltech101_val" in self.config.dataset.val:
            loaders["caltech101_val"] = DataLoader(
                Caltech101Dataset(
                    processor=self.processor,
                    **self.config.dataset.caltech101,
                ),
                **self.config.dataloader.caltech101_val,
                collate_fn=self._collate_fn,
            )
        return loaders

    # TODO: not sure if custom collate_fn works with pinned_memory in dataloader.
    # return type of processor is custom dict type, not sure if it works with pinned_memory
    def _collate_fn(self, batch):
        images, text = zip(*batch)
        return self.processor(
            images=images, text=text, padding=True, truncation=True, return_tensors="pt"
        )
