import pandas as pd
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np

def fetch_dataloaders(args):
    if args.input_data_format == "volumes":
        train_dataset = VolumeDataset(split="train", args=args)
        test_dataset = VolumeDataset(split="val", args=args)
    elif args.input_data_format in ["slices", "slices+volumes"]:
        train_dataset = SliceDataset(split="train", args=args)
        test_dataset = SliceDataset(split="val", args=args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = train_loader

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader, test_loader

def standardize_volume(data, n_channels=40):
    # Handle torch.Tensor inputs (preserve dtype and device)
    if isinstance(data, torch.Tensor):
        c = data.shape[0]
        if c < n_channels:
            deficit = n_channels - c
            pad_before = deficit // 2
            pad_after = deficit - pad_before
            # Create zero tensors with same dtype/device for padding
            pad_shape_before = (pad_before, ) + tuple(data.shape[1:])
            pad_shape_after = (pad_after, ) + tuple(data.shape[1:])
            pad_before_t = torch.zeros(pad_shape_before, dtype=data.dtype, device=data.device)
            pad_after_t = torch.zeros(pad_shape_after, dtype=data.dtype, device=data.device)
            data = torch.cat([pad_before_t, data, pad_after_t], dim=0)
        elif c > n_channels:
            excess = c - n_channels
            start_idx = excess // 2
            end_idx = start_idx + n_channels
            data = data[start_idx:end_idx, ...]
        return data

# TODO: look into csv data format
class VolumeDataset(Dataset):
    def __init__(self, split="train", args=None):
        self.args = args
        self.split = split
        if split == "train":
            file = args.train_file
        elif split == "val":
            file = args.val_file
        else:
            raise ValueError("Invalid split name")
        csv_data = pd.read_csv(file)
        logging.info(f"loading {len(csv_data)} {split} volumes")
        files = csv_data["file"].tolist()
        self.files = files
        self.labels = csv_data["meniscus_tear"].values
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_ = self.files[idx]
        label = self.labels[idx]
        
        vol_path = os.path.join(self.args.data_dir, f"singlecoil_{self.split}", f"{file_}_{self.args.inputs}.pt")
        # load as complex data
        volume = torch.load(vol_path)
        volume = standardize_volume(volume, n_channels=self.args.n_slices_per_volume)
        label = torch.tensor(label, dtype=torch.float32)

        # randomly rotate volume for data augmentation
        if self.split == "train":
            k = np.random.randint(0, 4)
            volume = torch.rot90(volume, k, [1, 2])

        return file_, volume, label

class SliceDataset(Dataset):
    def __init__(self, split="train", args=None, mask = None):
        self.args = args
        self.split = split
        if split == "train":
            file = args.train_file
        elif split == "val":
            file = args.val_file
        else:
            raise ValueError("Invalid split name")
        csv_data = pd.read_csv(file)
        logging.info(f"loading {len(csv_data)} {split} slices")

        logging.info(f"loading {len(csv_data)} {split} slices")
        self.csv_data = csv_data
        self.grouped_data = csv_data.groupby("file")
        self.files = csv_data["file"].unique().tolist()
        self.vol_paths = []

        for idx, file_ in enumerate(self.files):
            vol_path = os.path.join(self.args.data_dir, f"singlecoil_{self.split}", f"{file_}_{self.args.inputs}.pt")
            self.vol_paths.append(vol_path)

        self.mask = mask
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_ = self.files[idx]
        vol_path = self.vol_paths[idx]
        volume = torch.load(vol_path)
        volume = standardize_volume(volume, n_channels=self.args.n_slices_per_volume)
        
        labels = self.grouped_data.get_group(file_)["meniscus_tear"].values
        diff = volume.shape[0] - len(labels)
        if diff > 0:
            if diff % 2 == 0:
                pad_before = diff // 2
                pad_after = diff // 2
            else:
                pad_before = diff // 2
                pad_after = diff // 2 + 1
            labels = np.pad(labels, (pad_before, pad_after), mode='constant')
        label = torch.tensor(labels, dtype=torch.float32)

        # randomly rotate slice for data augmentation
        if self.split == "train":
            k = np.random.randint(0, 4)
            volume = torch.rot90(volume, k, [1, 2])
        
        vol_label = 1 if torch.sum(label) > 0 else 0
        vol_label = torch.tensor(vol_label, dtype=torch.float32)

        return file_, volume, label, vol_label
        
