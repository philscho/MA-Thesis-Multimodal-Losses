import os
import torch
from torchvision.datasets import Caltech101



class Caltech101Dataset(Caltech101):
    def __init__(self, processor=None, transform=None, download=True, root=os.path.expanduser("~/.cache")):
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
            # inputs = self.processor(images=image, 
            #                         #text=self.labels, 
            #                         padding='max_length', return_tensors="pt")
            # img = torch.squeeze(inputs['pixel_values'])
            # if self.transform:
            #     img = self.transform(img)
            # tokens = torch.squeeze(self.labels_tokenized['input_ids'])
            # token_type = torch.squeeze(self.labels_tokenized['token_type_ids'])
            # mask = torch.squeeze(self.labels_tokenized['attention_mask'])

            # return img, tokens, token_type, mask, label
            # try:
            image = torch.squeeze(self.processor(images=image, return_tensors="pt")['pixel_values'])
            # except ValueError:
            #     print (f"Issue found at {idx} with label {label}")
            #     import matplotlib.pyplot as plt
            #     plt.imshow(image)
            #     plt.savefig(f"./{idx}.jpg")
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

