from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNetDataset(Dataset):
    def __init__(self, 
                 split: str = 'train',
                 root='/home/data/ImageNet/ILSVRC/Data/CLS-LOC', 
                 transform=None,
                 processor=None,
                 classnames_path='src/data/datasets/imagenet-simple-labels.json',
                 synset_to_label_file='src/data/datasets/imagenet_synset_label_mapping.json',
                ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.root = os.path.join(root, 'train')
        self.split = split
        self.transform = transform
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.classes = json.load(open(classnames_path))
        self.synset_to_label = json.load(open(synset_to_label_file))

        if self.split == "train":
            # Assuming each subdirectory in root_dir is a class folder
            for label, class_dir in enumerate(sorted(os.listdir(self.root))):
                class_dir_path = os.path.join(self.root, class_dir)
                if os.path.isdir(class_dir_path):
                    for img_name in os.listdir(class_dir_path):
                        img_path = os.path.join(class_dir_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        elif self.split in ['1percent', '10percent']:
            with open(f"src/data/datasets/imagenet_{self.split}.txt") as f:
                files = f.readlines()
                self.image_paths = [os.path.join(self.root, f.split('_')[0], f.strip()) for f in files]
                self.labels = [int(self.synset_to_label[f.split('_')[0]][0]) for f in files]
        elif split == "100-10percent":
            with open(f"src/data/datasets/imagenet-100_classes.txt") as f:
                classes = f.read().strip().split()
            with open(f"src/data/datasets/imagenet_10percent.txt") as f:
                files = f.readlines()
                self.image_paths = [os.path.join(self.root, f.split('_')[0], f.strip()) 
                                    for f in files if f.split('_')[0] in classes]
                self.labels = [int(self.synset_to_label[f.split('_')[0]][0]) 
                               for f in files if f.split('_')[0] in classes]
        elif self.split == "100":
            self.classnames_100 = []
            with open(f"src/data/datasets/imagenet-100_classes.txt") as f:
                classes = f.read().strip().split()
                for imclass in classes:
                    class_dir_path = os.path.join(self.root, imclass)
                    if os.path.isdir(class_dir_path):
                        for img_name in os.listdir(class_dir_path):
                            img_path = os.path.join(class_dir_path, img_name)
                            self.image_paths.append(img_path)
                            label = int(self.synset_to_label[imclass][0])
                            self.labels.append(label)
                            self.classnames_100.append(self.classes[label])
            self.classes = self.classnames_100
        else:
            raise ValueError("Split must be one of ['train', '1percent', '10percent', '100'].")

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
                 classnames_path='src/data/datasets/imagenet-simple-labels.json'
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
    
if __name__ == "__main__":
    _ = ImageNetDataset(split="1_percent")