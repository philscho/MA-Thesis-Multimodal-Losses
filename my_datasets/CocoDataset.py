import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import os
import json

class CocoDataset(Dataset):
    def __init__(self, split_file, image_dir, processor, transform=None, simclr=False):
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.simclr = simclr
        #self.split = load_json(split_file)

        with open(split_file, 'r') as file:
            self.split = [json.loads(line.strip()) for line in file]

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, idx, caption_idx=None):
        sample = self.split[idx]
        img_filenames = sample['filename']
        img_filename = img_filenames[0] if isinstance(img_filenames, list) else img_filenames
        img_path = os.path.join(self.image_dir, img_filename)       
        # Load image
        image = Image.open(img_path).convert('RGB')
        # Load text
        if not caption_idx is None:
            i = caption_idx 
        else:
            i = random.randint(0, 4)
        caption = sample['sentences'][i]

        if self.processor is not None:
            inputs = self.processor(images=image, text=caption, padding='max_length', max_length=128, return_tensors="pt")
            if self.transform:
                inputs['pixel_values'] = self.transform(inputs['pixel_values'].to(torch.uint8))
                #img = self.transform(img.to(torch.uint8))
            for key in inputs:
                inputs[key] = torch.squeeze(inputs[key])
            return inputs
        else:
            if self.transform:
                image = self.transform(image)
            return image, caption
  

def load_json(filepath):
    json_objects = []
    with open(filepath, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            json_objects.append(json_object)
    return json_objects
