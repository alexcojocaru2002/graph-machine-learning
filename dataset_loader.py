from typing import Optional

import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob

import const


class DeepGlobeDataset(Dataset):
    def __init__(self, data_dir, class_rgb_values, img_size: Optional[tuple[int, int]] = const.IMG_SIZE):
        self.image_paths = sorted(glob(os.path.join(data_dir, '*_sat.jpg')))
        self.mask_paths  = sorted(glob(os.path.join(data_dir, '*_mask.png')))
        self.class_rgb_values = class_rgb_values
        self.img_size = img_size  # (W, H)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load the image and mask at index `idx`, resize them, and convert to tensors.
        """
        # --- Load & resize images ---
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.img_size is not None:
            img = img.resize(self.img_size, resample=Image.Resampling.BILINEAR)
        img_rgb = np.array(img) # needed for SLIC

        mask = Image.open(self.mask_paths[idx]).convert('RGB')
        if self.img_size is not None:
            mask = mask.resize(self.img_size, resample=Image.Resampling.NEAREST)

        # --- transform to tensors and preprocess ---
        prepoc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        img_t = prepoc(img)

        mask_np = np.array(mask)                   # (H,W,3), uint8
        mask_np = self.rgb_to_class(mask_np)  # (H,W), uint8
        mask_t = torch.from_numpy(mask_np).long()  # (H,W), int64

        return img_t, img_rgb, mask_t

    def rgb_to_class(self, mask):
        """
        Convert (H, W, 3) RGB mask to (H, W) class-index mask.
        """
        semantic_map = np.zeros(mask.shape[:2], dtype=np.uint8)
        for idx, color in enumerate(self.class_rgb_values):
            semantic_map[np.all(mask == color, axis=-1)] = idx
        return semantic_map
