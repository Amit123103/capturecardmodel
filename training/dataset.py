import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.augmentation import BusinessCardAugmentation

class BusinessCardDataset(Dataset):
    """
    Dataset loader supporting both Supervised (images + JSON labels)
    and Unsupervised (images only) modes.
    """
    def __init__(self, data_dir, tokenizer=None, split='train', mode='supervised'):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.split = split
        self.mode = mode
        self.augment = BusinessCardAugmentation()
        
        self.samples = []
        
        if self.mode == 'supervised':
            self.images_dir = os.path.join(data_dir, "images")
            self.labels_dir = os.path.join(data_dir, "labels")
            if os.path.exists(self.labels_dir):
                for file in os.listdir(self.labels_dir):
                    if file.endswith('.json'):
                        self.samples.append(file)
        elif self.mode == 'unsupervised':
            self.images_dir = data_dir
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append(os.path.join(root, file))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        if self.mode == 'supervised':
            label_file = self.samples[idx]
            with open(os.path.join(self.labels_dir, label_file), 'r') as f:
                data = json.load(f)
                
            img_name = data['image_file']
            img_path = os.path.join(self.images_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.augment(img, split=self.split)
            
            target_str = json.dumps(data['structured_data'], separators=(',', ':'))
            target_tokens = self.tokenizer.encode(target_str)
            
            return {
                'image': img_tensor,
                'target_tokens': torch.tensor(target_tokens, dtype=torch.long)
            }
            
        elif self.mode == 'unsupervised':
            img_path = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.augment(img, split=self.split)
            
            return {
                'image': img_tensor,
                'target_tokens': torch.zeros(1) 
            }
