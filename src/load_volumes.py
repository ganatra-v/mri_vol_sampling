import os
import pandas as pd
from tqdm import tqdm
import logging
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

def standardize_volume(data, args):
    if data.shape[0] < args.n_channels:
        deficit = args.n_channels - data.shape[0]
        pad_before = deficit // 2
        pad_after = deficit - pad_before
        data = np.pad(data, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
    elif data.shape[0] > args.n_channels:
        excess = data.shape[0] - args.n_channels
        start_idx = excess // 2
        end_idx = start_idx + args.n_channels
        data = data[start_idx:end_idx, :, :]
    return data

def fetch_dataloaders(args):
    train_dataset = VolumeDataset(split="train", args=args)
    val_dataset = VolumeDataset(split="val", args=args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader

class VolumeDataset(Dataset):
    def __init__(self, split="train", args=None):
        self.args = args
        self.split = split
        if split == "train":
            file = args.train_file
        else:
            file = args.val_file
        csv_data = pd.read_csv(file)
        logging.info(f"loading {len(csv_data)} train volumes")
        files = csv_data["file"].tolist()
        self.files = files
        self.labels = csv_data["meniscus_tear"].values


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        file_ = self.files[idx]
        vol_path = os.path.join(self.args.data_dir, f"singlecoil_{self.split}", f"{file_}.h5")
        with h5py.File(vol_path, "r") as f:
            data = f[self.args.inputs][()]
            data = standardize_volume(data, self.args)
        
        volume = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # randomly rotate volume for data augmentation
        if self.split == "train":
            k = np.random.randint(0, 4)
            volume = torch.rot90(volume, k, [1, 2])

        return volume, label