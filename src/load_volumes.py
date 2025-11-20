import os
import pandas as pd
from tqdm import tqdm
import logging
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

def standardize_volume(data, n_channels=40):
    if data.shape[0] < n_channels:
        deficit = n_channels - data.shape[0]
        pad_before = deficit // 2
        pad_after = deficit - pad_before
        data = np.pad(data, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
    elif data.shape[0] > n_channels:
        excess = data.shape[0] - n_channels
        start_idx = excess // 2
        end_idx = start_idx + n_channels
        data = data[start_idx:end_idx, :, :]
    return data

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

class VolumeDataset(Dataset):
    def __init__(self, split="train", args=None):
        self.args = args
        self.split = split
        if split == "train":
            file = args.train_file
        elif split == "val":
            file = args.val_file
        elif split == "test":
            file = args.test_file
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
        vol_path = os.path.join(self.args.data_dir, f"{file_}.h5")
        with h5py.File(vol_path, "r") as f:
            data = f[self.args.inputs][()]
            data = standardize_volume(data, self.args)
        
        volume = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

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
        elif split == "test":
            file = args.test_file
        else:
            raise ValueError("Invalid split name")
        csv_data = pd.read_csv(file)
        logging.info(f"loading {len(csv_data)} {split} slices")
        self.csv_data = csv_data
        self.grouped_data = csv_data.groupby("file")
        self.files = csv_data["file"].unique().tolist()

        self.mask = mask
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_ = self.files[idx]
        vol_path = os.path.join(self.args.data_dir, f"singlecoil_{self.args.split}/{file_}.h5")
        with h5py.File(vol_path, "r") as f:
            data = f[self.args.inputs][()]
            if self.args.inputs == "kspace" and self.args.input_project == "ifft_preprocess":
                # apply ifft to k-space data
                data = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(data, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
                data = np.abs(data)
                # crop the central (320, 320) region
                center_x, center_y = data.shape[-2] // 2, data.shape[-1] // 2
                crop_size = 320
                data = data[
                    :,
                    center_x - crop_size // 2 : center_x + crop_size // 2,
                    center_y - crop_size // 2 : center_y + crop_size // 2,
                ]

            data = standardize_volume(data, n_channels=50)            

            if self.mask is None:
                # randomly sampling slices based on vol_sampling_fraction            
                n_slices = data.shape[0]
                n_sampled_slices = max(1, int(n_slices * self.args.vol_sampling_fraction))
                sampled_indices = np.sort(np.random.choice(n_slices, n_sampled_slices, replace=False))
                # zero out non-sampled slices
                mask = np.zeros(n_slices, dtype=bool)
                mask[sampled_indices] = True
            else:
                mask = self.mask

            data[~mask, :, :] = 0

    

        
        volume = torch.tensor(data, dtype=torch.float32)
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
        
        labels[~mask] = 0  # set labels of non-sampled slices to 0
        label = torch.tensor(labels, dtype=torch.float32)

        # randomly rotate volume for data augmentation
        if self.split == "train":
            k = np.random.randint(0, 4)
            volume = torch.rot90(volume, k, [1, 2])
        
        assert volume.shape[0] == label.shape[0], "Number of slices and labels must match"
        volume_label = 1.0 if label.sum() > 0 else 0.0
        volume_label = torch.tensor(volume_label, dtype=torch.float32)
        return file_, volume, label, volume_label