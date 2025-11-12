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
        if split == "train":
            file = args.train_file
        else:
            file = args.val_file
        data = pd.read_csv(file)
        logging.info(f"loading {len(data)} train volumes")
        files = data["file"].tolist()
        self.volumes = []
        for idx, file_ in tqdm(enumerate(files)):
            vol_path = os.path.join(args.data_dir, "singlecoil_train", f"{file_}.h5")
            with h5py.File(vol_path, "r") as f:
                data = f[args.inputs][()]
                data = standardize_volume(data, args)
                self.volumes.append(data)
        self.labels = data["meniscus_tear"].values
        logging.info(f"loaded {len(self.volumes)} volumes")


    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, idx):
        volume = self.volumes[idx]
        label = self.labels[idx]
        volume = torch.tensor(volume)
        return volume, label