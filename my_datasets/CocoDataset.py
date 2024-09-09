import json
import random
import os
from typing import Any, List, Dict, Union, Tuple

from PIL import Image
from transformers import VisionTextDualEncoderProcessor
import torch
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(
            self, 
            split_file: str, 
            image_dir: str, 
            processor: VisionTextDualEncoderProcessor, 
            transform: torch.nn.Module = None, 
            simclr: bool = False
    ) -> None:
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.simclr = simclr
        self.split: List[Dict[str, Union[str, List[str]]]] = []

        with open(split_file, 'r') as file:
            self.split = [json.loads(line.strip()) for line in file]

    def __len__(self) -> int:
        return len(self.split)
    
    def __getitem__(self, 
                    idx: int,
                    caption_idx: Union[int, None] = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[Image.Image, str]]:
        example = self.split[idx]
        img_filenames = example['filename']
        img_filename = img_filenames[0] if isinstance(img_filenames, list) else img_filenames
        img_path = os.path.join(self.image_dir, img_filename)       
        image = Image.open(img_path).convert('RGB')
        
        if not caption_idx is None:
            i = caption_idx 
        else:
            i = random.randint(0, 4)
        caption = example['sentences'][i]

        if self.processor is not None:
            inputs = self.processor(images=image, text=caption, padding='max_length', max_length=128, return_tensors="pt")
            if self.transform:
                inputs['pixel_values'] = self.transform(inputs['pixel_values'].to(torch.uint8))
            for key in inputs:
                inputs[key] = torch.squeeze(inputs[key])
            return inputs
        # Return value for use of processor in collate_fn in DataLoader
        else:
            if self.transform:
                image = self.transform(image)
            return image, caption
  

def load_json(filepath: str) -> List[Dict[str, Union[str, List[str]]]]:
    json_objects = []
    with open(filepath, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            json_objects.append(json_object)
    return json_objects
