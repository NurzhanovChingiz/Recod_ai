import os
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

class ForgerySegDataset(Dataset):
    def __init__(self, auth_paths, forg_paths, mask_dir, transform=None):
        self.samples = []
        
        # Add forged samples with masks
        for p in forg_paths:
            m = os.path.join(mask_dir, Path(p).stem + ".npy")
            if os.path.exists(m):
                self.samples.append((p, m))
        
        # Add authentic samples with null masks
        for p in auth_paths:
            self.samples.append((p, None))
            
        self.transform = transform
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        from config import CFG
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Load or create mask
        if mask_path is None:
            mask = np.zeros((h, w), np.uint8)    
        else:
            m = np.load(mask_path)
            if m.ndim == 3:
                m = np.max(m, axis=0)
            mask = (m > 0).astype(np.uint8)
        
        # Resize both image and mask to consistent size
        # img_resized = img.resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BILINEAR)
        mask = cv2.resize(mask, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        
        mask_tensor = torch.from_numpy(mask[None, ...].astype(np.float32))
        
        # Apply transforms to tensors (if any)
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(np.array(img, np.float32) / 255.0).permute(2, 0, 1)
            
        return img_tensor, mask_tensor