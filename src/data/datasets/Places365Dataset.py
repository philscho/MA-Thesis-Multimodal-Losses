import torch
from torch.utils.data import Dataset
import torchvision
import random
import numpy as np

class Places365Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, fraction=1.0, seed=42, transform=None):
        super(Places365Dataset, self).__init__(root, transform=transform)
        self.fraction = fraction
        self.seed = seed
        self.indices = self._get_balanced_indices()

    def _get_balanced_indices(self):
        if self.seed is not None:
            random.seed(self.seed)
        class_indices = {i: [] for i in range(len(self.classes))}
        for idx, (_, label) in enumerate(self.samples):
            class_indices[label].append(idx)
        
        num_samples_per_class = int(np.floor(len(self.samples) * self.fraction / len(self.classes)))
        balanced_indices = []
        for indices in class_indices.values():
            balanced_indices.extend(random.sample(indices, min(len(indices), num_samples_per_class)))
        
        random.shuffle(balanced_indices)
        return balanced_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return super(Places365Dataset, self).__getitem__(actual_idx)

# Example usage:
# dataset = Places365Dataset(root='/path/to/places365', fraction=0.5, seed=42, transform=your_transform)