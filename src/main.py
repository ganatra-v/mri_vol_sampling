import argparse
import random
import numpy as np
import torch
import os
import logging
from load_volumes import fetch_dataloaders
from models.vol_cls_model import VolClsModel
from models.slice_model import SliceModel

parser = argparse.ArgumentParser(description="MRI Volume Sampling")
parser.add_argument("--dataset", type=str, choices=["knee"], required=True, help="Dataset to use. Currently only 'knee' is supported.")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
parser.add_argument("--train_file", type=str, required=True, help="CSV file listing training files.")
parser.add_argument("--val_file", type=str, required=True, help="CSV file listing validation files.")
parser.add_argument("--test_file", type=str, required=True, help="CSV file listing test files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")

parser.add_argument("--inputs", type=str, choices=["kspace", "reconstruction_esc", "reconstruction_rss"], default="kspace", help="Type of input data.")
parser.add_argument("--input_project", type=str, choices=["none", "ifft_preprocess", "kspace_crop"], default="none", help="Preprocessing to apply to input data.")
parser.add_argument("--input_data_format", type=str, choices=["slices", "volumes", "slices+volumes"], default="volumes", help="Format of input data.")
parser.add_argument("--slice_sampling_fraction", type=float, default=0.1, help="Fraction of k-space to sample.")
parser.add_argument("--vol_sampling_fraction", type=float, default=0.5, help="Fraction of volumes to sample.")
parser.add_argument("--n_channels", type=int, default=1, help="Number of channels in the input data.")
parser.add_argument("--save_topk_indices", action="store_true", help="Save top-k slice indices based on attention weights.")
parser.add_argument("--topk", type=int, default=5, help="Number of top slices to save based on attention weights.")
parser.add_argument("--n_masks_eval", type=int, default=150, help="Number of random masks to sample during evaluation.")

parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

parser.add_argument("--arch", type=str, default="resnet18", help="Model architecture to use.")
parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for the model.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
parser.add_argument("--slice_loss_lam", type=float, default=1.0, help="Weight for slice-level loss.")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")
parser.add_argument("--eval_interval", type=int, default=5, help="Interval (in epochs) for evaluation on validation set.")

parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the checkpoint to resume from.")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_outdir(args):
    train_details = f"{args.dataset}_{args.inputs}_{args.input_data_format}_{args.input_project if args.input_project != 'none' else ''}vol_sampling_{args.vol_sampling_fraction}_slice_sampling_{args.slice_sampling_fraction}_slice_loss_lam_{args.slice_loss_lam}/"
    outdir = f"{args.output_dir}/{train_details}/"
    outdir = f"{outdir}/{args.arch}_lr_{args.learning_rate}_bs_{args.batch_size}_{args.epochs}_epochs_wd_{args.weight_decay}/"
    os.makedirs(outdir, exist_ok=True)
    return outdir

def sample_masks(n_masks, vol_size = 50, sampling_fraction=0.5):
    masks = []
    for _ in range(n_masks):
        n_slices = vol_size
        n_sampled_slices = max(1, int(n_slices * sampling_fraction))
        sampled_indices = np.sort(np.random.choice(n_slices, n_sampled_slices, replace=False))
        # zero out non-sampled slices
        mask = np.zeros(n_slices, dtype=bool)
        mask[sampled_indices] = True
        masks.append(mask)
    return masks

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
    logging.info(vars(args))

    # fetch dataloaders
    train_loader, val_loader, test_loader = fetch_dataloaders(args)

    if args.input_data_format == "volumes":
        model = VolClsModel(args)
    elif args.input_data_format in ["slices", "slices+volumes"]:
        model = SliceModel(args)

    model = model.cuda() if torch.cuda.is_available() else model

    if args.resume and args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path))
        logging.info(f"Resumed training from checkpoint: {args.checkpoint_path}")

    model.train_model(train_loader, val_loader)
    logging.info("loading best model for evaluation..........")
    model.load_state_dict(torch.load(os.path.join(args.outdir, "best_model.pth")))

    logging.info("Val set .................................................")
    model.eval()
    model.eval_model(val_loader, save_topk_slices=args.save_topk_indices, save_preds=True)

    logging.info("Test set ................................................")
    model.eval()
    model.eval_model(test_loader, save_topk_slices=args.save_topk_indices, save_preds=True)
    logging.info("Training and evaluation completed.")

    if args.vol_sampling_fraction < 1.0:
        # sample multiple masks and compute val performance for each
        logging.info("Evaluating multiple volume sampling masks ................")
        n_masks = args.n_masks_eval
        masks = sample_masks(n_masks, vol_size=50, sampling_fraction=args.vol_sampling_fraction)
        val_accuracies = []
        for i, mask in enumerate(masks):
            _, val_loader_masked = fetch_dataloaders(args, mask=mask)
            logging.info(f"Evaluating mask {i+1}/{n_masks} ................")
            val_acc = model.eval_model(val_loader_masked, save_topk_slices=False, save_preds=False)
            val_accuracies.append(val_acc)

        max_acc_idx = np.argmax(val_accuracies)
        logging.info(f"Best mask index: {max_acc_idx} with val accuracy: {val_accuracies[max_acc_idx]}")
        best_mask = masks[max_acc_idx]
        # evaluate on test set with best mask
        _, test_loader_best_mask = fetch_dataloaders(args, mask=best_mask)
        logging.info("Evaluating best mask on test set ....................")
        model.eval_model(test_loader_best_mask, save_topk_slices=False, save_preds=True)