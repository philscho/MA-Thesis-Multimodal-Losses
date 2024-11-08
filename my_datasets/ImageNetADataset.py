import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNetADataset(Dataset):
    def __init__(self, 
                 root_dir='/home/data/phisch/imagenet-a/val', 
                 transform=None, 
                 processor=None,
                 classnames_path='my_datasets/imagenet-a_class-names.json'
                 ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
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
