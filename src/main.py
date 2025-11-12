import argparse
import random
import numpy as np
import torch
import os
import logging
from load_volumes import fetch_dataloaders
from models.vol_cls_model import VolClsModel

parser = argparse.ArgumentParser(description="MRI Volume Sampling")
parser.add_argument("--dataset", type=str, choices=["knee"], required=True, help="Dataset to use. Currently only 'knee' is supported.")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
parser.add_argument("--train_file", type=str, required=True, help="CSV file listing training files.")
parser.add_argument("--val_file", type=str, required=True, help="CSV file listing validation files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")

parser.add_argument("--inputs", type=str, choices=["kspace", "reconstruction_esc"], default="kspace", help="Type of input data.")
parser.add_argument("--slice_sampling_fraction", type=float, default=0.1, help="Fraction of k-space to sample.")
parser.add_argument("--vol_sampling_fraction", type=float, default=0.5, help="Fraction of volumes to sample.")
parser.add_argument("--n_channels", type=int, default=1, help="Number of channels in the input data.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

parser.add_argument("--arch", type=str, default="resnet18", help="Model architecture to use.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_outdir(args):
    train_details = f"{args.dataset}_{args.inputs}_vol_sampling_{args.vol_sampling_fraction}_slice_sampling_{args.slice_sampling_fraction}"
    outdir = f"{args.output_dir}/{train_details}/"
    outdir = f"{outdir}/{args.arch}_lr_{args.learning_rate}_bs_{args.batch_size}_{args.epochs}_epochs_wd_{args.weight_decay}/"
    os.makedirs(outdir, exist_ok=True)
    return outdir


if __name__ == "__main__":
    args = parser.parse_args()
    print(vars(args))

    # seed everything for reproducibility
    seed_everything(args.seed)

    # setup outdir
    args.outdir = setup_outdir(args)

    # setup logging
    logging.basicConfig(
        filename=os.path.join(args.outdir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # fetch dataloaders
    train_loader, val_loader = fetch_dataloaders(args)

    model = VolClsModel(args)
    model = model.cuda() if torch.cuda.is_available() else model

    model.train_model(train_loader)
    model.evaluate_model(val_loader)