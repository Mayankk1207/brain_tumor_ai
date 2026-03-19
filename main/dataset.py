import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BrainDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = "/home/mayank/CV/main/data_processed/train/images"
        self.mask_dir = "/home/mayank/CV/main/data_processed/train/masks"

        self.images = sorted(os.listdir(img_dir))
        self.masks  = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir, self.images[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.masks[idx]))

        # Normalize safety
        if img.max() > 1:
            img = img / (img.max() + 1e-8)

        # Ensure label fix
        mask[mask == 4] = 3

        img = torch.tensor(img, dtype=torch.float32)   # (4,H,W)
        mask = torch.tensor(mask, dtype=torch.long)    # (H,W)

        return img, mask


def get_loaders(train_img, train_mask, val_img, val_mask, batch_size=4):
    train_dataset = BrainDataset(train_img, train_mask)
    val_dataset   = BrainDataset(val_img, val_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader