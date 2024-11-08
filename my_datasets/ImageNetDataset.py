import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNetDataset(Dataset):
    def __init__(self, 
                 split='val',
                 root_dir='/home/data/ImageNet/ILSVRC/Data/CLS-LOC/', 
                 transform=None,
                 processor=None,
                 classnames_path='my_datasets/imagenet-simple-labels.json'
                ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.split = split
        if self.split != 'val':
            raise ValueError("Only validation split is supported for now.")
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.classnames = json.load(open(classnames_path))
        
        # Assuming each subdirectory in root_dir is a class folder
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for img_name in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")

        if self.processor:
            image = torch.squeeze(self.processor(images=image, return_tensors="pt")['pixel_values'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class ImageNetValDataset(Dataset):
    def __init__(self, 
                 img_dir='/home/data/ImageNet/ILSVRC/Data/CLS-LOC/val', 
                 labels_path='/home/data/ImageNet/imagenet_val_labels.json', 
                 transform=None,
                 processor=None,
                 classnames_path='my_datasets/imagenet-simple-labels.json'
                 ):
        self.img_dir = img_dir
        self.labels = json.load(open(labels_path))
        self.image_ids = list(self.labels.keys())
        self.transform = transform
        self.classnames = json.load(open(classnames_path))
        self.processor = processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.JPEG")
        image = Image.open(img_path).convert("RGB")
        label = self.labels[image_id]["class_index"]  # Use the class index from the JSON file

        if self.processor:
            image = torch.squeeze(self.processor(images=image, return_tensors="pt")['pixel_values'])
        if self.transform:
            image = self.transform(image)

        return image, label