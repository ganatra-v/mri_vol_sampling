import json
import argparse
import h5py
import os

def fetch_num_slices(in_dir):
    files = os.listdir(in_dir)
    num_slices = {}
    for fname in files:
        if fname.endswith('.h5'):
            with h5py.File(os.path.join(in_dir, fname), 'r') as hf:
                n_slices = hf['reconstruction_esc'].shape[0]
                num_slices[fname] = n_slices
    return num_slices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file.")
    args = parser.parse_args()
    num_slices = fetch_num_slices(args.data_dir)
    with open(args.output_file, 'w') as f:
        json.dump(num_slices, f)