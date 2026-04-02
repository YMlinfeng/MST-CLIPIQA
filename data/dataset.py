import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .transforms import get_clip_transforms

class AGIQADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
                               Expected columns: 'image_name', 'prompt', 'mos'
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else get_clip_transforms()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx]['image_name']))
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        prompt = str(self.data_frame.iloc[idx]['prompt'])
        mos = float(self.data_frame.iloc[idx]['mos'])
        
        return {
            'image': image,
            'prompt': prompt,
            'mos': mos
        }
