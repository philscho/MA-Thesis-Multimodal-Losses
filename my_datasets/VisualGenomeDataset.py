import h5py
from pathlib import Path
import pandas as pd
import io 
import torch
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

class VisualGenomeDataset:
    def __init__(self, root, processor=None, transform=None, num_views=1, use_llava_split=False):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError("Root must be a dir.")
        
        if use_llava_split:
            self.mapper = pd.read_parquet(self.root/'mapper_LLaVA_clean.parquet')
        else:
            self.mapper = pd.read_parquet(self.root / 'mapper.parquet')
        self.h5_file = h5py.File(self.root / 'vg.h5', 'r')
        
        self.transform = transform
        self.num_views = num_views
        self.processor = processor
        
    def __len__(self):
        return len(self.mapper)
    
    def __getitem__(self, idx):
        dp = self.mapper.iloc[idx]
        
        raw_image = self.h5_file.get("images")[dp.h5_index]
        image = Image.open(io.BytesIO(raw_image))
        caption = np.random.choice(dp.caption)
        
        if self.processor is not None:
            inputs = self.processor(images=image, text=caption, padding='max_length', return_tensors="pt")
            image = torch.squeeze(inputs['pixel_values'])
            if self.transform is not None:
                image = self.transform(image)
            tokens = torch.squeeze(inputs['input_ids'])
            token_type = torch.squeeze(inputs['token_type_ids'])
            mask = torch.squeeze(inputs['attention_mask'])

            return image, tokens, token_type, mask
        else:
            if self.transform:
                images = [self.transform(image) for _ in range(self.num_views)]
            else:
                images = [image]
            return images, caption
