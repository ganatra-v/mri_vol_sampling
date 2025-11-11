import os
import pandas as pd
from tqdm import tqdm
import logging
import h5py

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
            train_volumes.append(data)
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
            val_volumes.append(data)
    logging.info(f"loaded {len(val_volumes)} val volumes")
    return train_volumes, val_volumes

def fetch_dataloaders(args):
    train_volumes, val_volumes = load_volumes(args)
    # TODO: implement DataLoader creation with volume and slice sampling
    pass