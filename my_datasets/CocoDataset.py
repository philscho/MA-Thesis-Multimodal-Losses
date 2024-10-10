import json
import random
import os
from typing import Any, List, Dict, Union, Tuple

from PIL import Image
from transformers import VisionTextDualEncoderProcessor
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CocoDataset(Dataset):
    def __init__(
            self, 
            root: str = '/home/data/COCOcaptions',
            split: str = 'train',
            processor: VisionTextDualEncoderProcessor = None, 
            transform: torch.nn.Module = None,
            num_views: int = 1, 
    ) -> None:
        self.root = root
        self.split = split
        self.dir = os.path.join(root, f"{split}2014")
        self.processor = processor
        self.transform = transform
        self.num_views = num_views
        with open(f"my_datasets/coco_karpathy_{split}.json", 'r') as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, 
                    idx: int,
                    caption_idx: int = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[Image.Image, str]]:
        example = self.data[idx]
        img_filenames = example['filename']
        img_filename = img_filenames[0] if isinstance(img_filenames, list) else img_filenames
        img_path = os.path.join(self.dir, img_filename)       
        image = Image.open(img_path).convert('RGB')
        
        i = caption_idx if caption_idx else random.randint(0, 4)
        caption = example['sentences'][i]

        if self.processor:
            inputs = self.processor(images=image, text=caption, 
                                    padding='max_length', max_length=128, 
                                    return_tensors="pt"
                                    )
            if self.transform:
                inputs['pixel_values'] = self.transform(inputs['pixel_values'].to(torch.uint8))
            for key in inputs:
                inputs[key] = torch.squeeze(inputs[key])
            return inputs
        # Return value for use of processor in collate_fn in DataLoader
        else:
            if self.transform:
                images = [self.transform(image) for _ in range(self.num_views)]
            else:
                images = [image]
            return images, caption
  

def load_json(filepath: str) -> List[Dict[str, Union[str, List[str]]]]:
    json_objects = []
    with open(filepath, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            json_objects.append(json_object)
    return json_objects
