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


def load_volumes(args):
    trainfile = args.train_file
    traindata = pd.read_csv(trainfile)
    logging.info(f"loading {len(traindata)} train volumes")
    train_files = traindata["file"].tolist()
    train_volumes = []
    for idx, file_ in tqdm(enumerate(train_files)):
        vol_path = os.path.join(args.data_dir, "singlecoil_train", f"{file_}.h5")
        with h5py.File(vol_path, "r") as f:
            data = f[args.inputs][()]
            data = standardize_volume(data, args)
            train_volumes.append(data)
    trainlabels = traindata["meniscus_tear"].values
    logging.info(f"loaded {len(train_volumes)} train volumes")

    valfile = args.val_file
    valdata = pd.read_csv(valfile)
    logging.info(f"loading {len(valdata)} val volumes")
    val_files = valdata["file"].tolist()
    val_volumes = []

    for idx, file_ in tqdm(enumerate(val_files)):
        vol_path = os.path.join(args.data_dir, "singlecoil_val", f"{file_}.h5")
        with h5py.File(vol_path, "r") as f:
            data = f[args.inputs][()]
            data = standardize_volume(data, args)
            val_volumes.append(data)
    logging.info(f"loaded {len(val_volumes)} val volumes")
    val_labels = valdata["meniscus_tear"].values
    logging.info("saving processed volumes to disk")
    np.savez_compressed(
        os.path.join(args.outdir, "train_volumes.npz"),
        volumes=train_volumes,
        labels=trainlabels
    )
    logging.info("saved processed train volumes, now saving val volumes")
    np.savez_compressed(
        os.path.join(args.outdir, "val_volumes.npz"),
        volumes=val_volumes,
        labels=val_labels
    )
    logging.info("saved processed val volumes")

def fetch_dataloaders(args):
    load_volumes(args)
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
        self.volumes = np.load(
            os.path.join(
                args.outdir,
                f"{split}_volumes.npz"
            ),
            allow_pickle=True
        )["volumes"]
        self.labels = np.load(
            os.path.join(
                args.outdir,
                f"{split}_volumes.npz"
            ),
            allow_pickle=True
        )["labels"]

    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, idx):
        volume = self.volumes[idx]
        label = self.labels[idx]
        volume = torch.tensor(volume)
        return volume, label