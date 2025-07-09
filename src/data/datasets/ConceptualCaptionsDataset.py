import h5py
from pathlib import Path
import pandas as pd
import io 
import torch
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

class ConceptualCaptionsDataset:
    def __init__(self, root, processor=None, transform=None, num_views=1, use_llava_split=False):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError("Root must be a dir.")
        
        if use_llava_split:
            self.mapper = pd.read_parquet('/home/data/CC3M_LLaVA/mapper_LLaVA_clean.parquet')
        else:
            self.mapper = pd.read_parquet(self.root / 'mapper.parquet')
        self.linker = h5py.File(self.root / 'linker.h5', 'r')
        
        self.transform = transform
        self.num_views = num_views
        self.processor = processor
        
    def __len__(self):
        return len(self.mapper)
    
    def __getitem__(self, idx):
        dp = self.mapper.iloc[idx]
        
        raw_image = self.linker.get(f"CC3m_{dp.shard}").get('images')[dp.h5_index]
        image = Image.open(io.BytesIO(raw_image)).convert('RGB') #dataset has a few grayscale images, converting them        
        caption = dp.caption
        
        if self.processor is not None:
            inputs = self.processor(images=image, text=caption, padding='max_length', return_tensors="pt")
            pixel_values = torch.squeeze(inputs['pixel_values'])
            if self.transform is not None:
                pixel_values = self.transform(pixel_values)
            input_ids = torch.squeeze(inputs['input_ids'])
            token_type_ids = torch.squeeze(inputs['token_type_ids'])
            attention_mask = torch.squeeze(inputs['attention_mask'])

            return pixel_values, input_ids, token_type_ids, attention_mask
        else:
            if self.transform:
                images = [self.transform(image) for _ in range(self.num_views)]
            else:
                images = [image]
            return images, caption
