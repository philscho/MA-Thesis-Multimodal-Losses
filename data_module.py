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
            self.train_datasets = self._get_train_datasets()
            self.val_datasets = self._get_val_datasets()
        else:
            raise NotImplementedError(f"Stage {stage} not implemented")

    def _get_train_datasets(self):
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

    def _get_val_datasets(self):
        val_datasets = {}
        if "coco_val" in self.config.dataset.val:
            val_datasets["coco_val"] = CocoDataset(
                self.config.dataset.coco.split_val,
                os.path.join(self.config.dataset.coco.data_dir, "val2014"),
                processor=None,
            )
        if "cifar10_val" in self.config.dataset.val:
            val_datasets["cifar10_val"] = Cifar10Dataset(
                processor=self.processor,
                **self.config.dataset.cifar10,
            )
        if "caltech101_val" in self.config.dataset.val:
            val_datasets["caltech101_val"] = Caltech101Dataset(
                processor=self.processor,
                **self.config.dataset.caltech101,
            )
        return val_datasets

    def train_dataloader(self):
        return DataLoader(self.train_datasets, **self.config.dataloader.train, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return {
            key: DataLoader(dataset, **getattr(self.config.dataloader, key), collate_fn=self._collate_fn)
            for key, dataset in self.val_datasets.items()
        }

    # TODO: not sure if custom collate_fn works with pinned_memory in dataloader.
    # return type of processor is custom dict type, not sure if it works with pinned_memory
    def _collate_fn(self, batch):
        images, text = zip(*batch)
        return self.processor(
            images=images, text=text, padding=True, truncation=True, return_tensors="pt"
        )
