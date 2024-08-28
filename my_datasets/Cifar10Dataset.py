import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import os
#from transformers import AutoImageProcessor, AutoTokenizer, VisionTextDualEncoderProcessor

class Cifar10Dataset(CIFAR10):
    def __init__(self, processor=None, transform=None, download=False, root=os.path.expanduser("~/.cache")):
        super().__init__(root=root, download=download, train=False)
        #self.dataset = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
        self.processor = processor
        self.transform = transform
        self.classnames = self.classes
        self.labels = [f"a photo of a {c}" for c in self.classes]
        if self.processor:
            self.labels_tokenized = self.processor(text=self.labels, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

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

            image = torch.squeeze(self.processor(images=image, return_tensors="pt")['pixel_values'])
            return image, label
        else:
            if self.transform:
                image = self.transform(image)
            return image, label




if __name__ == "__main__":
    from transformers import AutoImageProcessor, AutoTokenizer, VisionTextDualEncoderProcessor
    
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', max_length=64, use_fast=False)
    processor = VisionTextDualEncoderProcessor(image_processor=image_processor, tokenizer=tokenizer)

    cifar10 = Cifar10Dataset()
    print(cifar10.classnames)
    
    # print(len(cifar10))
    # img, tokens, token_type, mask, label = cifar10[100]
    # print(cifar10.labels)
    # print(cifar10.labels_tokenized)
    # print(img.shape, tokens.shape, token_type.shape, mask.shape)
    # print(label)
    # print(cifar10.targets)
