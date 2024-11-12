import os
import torch
from torchvision.datasets import Caltech101



class Caltech101Dataset(Caltech101):
    def __init__(self, processor=None, transform=None, download=False, root=os.path.expanduser("~/.cache")):
        super().__init__(root=root, download=download)
        #self.dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
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



if __name__ =='__main__':
    
    from transformers import AutoImageProcessor, AutoTokenizer, VisionTextDualEncoderProcessor 
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('default')
    
    
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', max_length=64, use_fast=False)
    processor = VisionTextDualEncoderProcessor(image_processor=image_processor, tokenizer=tokenizer)

    caltech_dataset = Caltech101Dataset(processor=processor)
    print(len(caltech_dataset))
    img, label = caltech_dataset[100]
    print ('--'*20)
    print(caltech_dataset.labels)
    print ('--'*20)
    print(caltech_dataset.labels_tokenized)
    print ('--'*20)
    print(img.shape)
    print ('--'*20)
    print(label)
    print ('--'*20)
    # print(caltech_dataset.targets)

