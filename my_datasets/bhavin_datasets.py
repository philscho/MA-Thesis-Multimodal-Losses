import os
import torch
from torchvision.datasets import SBU



class SBU(SBU):
    def __init__(self, processor=None, transform=None, download=False, root=os.path.expanduser("~/.cache")):
        super().__init__(root=root, download=download)
        
        self.processor = processor
        self.transform = transform
        #TODO: Clean up label names
        self.labels = [f"a photo of a {c}" for c in self.categories]
        if self.processor:
            self.labels_tokenized = self.processor(text=self.labels, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image = image.convert('RGB')

        if self.processor is not None:
            image = torch.squeeze(self.processor(images=image, return_tensors="pt")['pixel_values'])

            return image, label
        else:
            if self.transform:
                image = self.transform(image)
            return image, label

