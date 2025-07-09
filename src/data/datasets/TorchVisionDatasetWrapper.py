import json
import torch

from ..utils import get_dataset_classnames

class TorchVisionDatasetWrapper:
    def __init__(self, baseclass, processor=None, transform=None, label_as_caption=False, caption_template="{}"):
        self.baseclass = baseclass
        self.processor = processor
        self.transform = transform
        self.classnames = get_dataset_classnames(baseclass)
        self.num_classes = len(self.classnames)
        self.label_as_caption = label_as_caption
        self.caption_template = caption_template

    def __getitem__(self, idx):
        image, label = self.baseclass.__getitem__(idx)
        image = image.convert("RGB")

        if self.processor is not None:
            image = torch.squeeze(
                self.processor(images=image, return_tensors="pt")["pixel_values"]
            )
        if self.transform:
            image = self.transform(image)
        
        if self.label_as_caption:
            classname = self.classnames[label] # convert int label to string caption
            caption = self.caption_template.format(classname)
            label = torch.squeeze(
                self.processor(text=caption, padding=True, truncation=True, return_tensors="pt")["input_ids"]
            )

        return image, label

    def __len__(self):
        return self.baseclass.__len__()



if __name__ == "main":
    pass
