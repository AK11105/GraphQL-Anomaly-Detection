# ml/training/dataloader.py
"""
Data loader utilities for the ML training scripts.

Expected dataset files inside dataset_dir:
  - train_windows.npy, train_labels.npy
  - val_windows.npy, val_labels.npy
  - test_windows.npy, test_labels.npy

Also supports .npz files with keys 'windows' and 'labels'.

Windows shape: (N, seq_len, feature_dim)
Labels shape: (N,) or (N,1)  - values 0 or 1
"""

import os
import numpy as np
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

def _load_np_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        # Expect keys 'windows' and 'labels' or fallback to first two arrays
        if "windows" in data and "labels" in data:
            return data["windows"], data["labels"]
        keys = list(data.keys())
        if len(keys) >= 2:
            return data[keys[0]], data[keys[1]]
        raise ValueError(f"npz file {path} missing expected arrays")
    elif ext == ".npy":
        # caller must know which .npy this is
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

class SequenceDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """
        windows: ndarray (N, seq_len, feature_dim)
        labels: ndarray (N,) or (N,1)
        """
        assert isinstance(windows, np.ndarray)
        assert isinstance(labels, np.ndarray)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        assert windows.shape[0] == labels.shape[0]
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

def load_split(dataset_dir: str, split: str):
    """
    Load windows and labels for a split string: 'train', 'val', 'test'.
    """
    split = split.lower()
    # Try common filenames
    candidates = [
        (os.path.join(dataset_dir, f"{split}_windows.npy"), os.path.join(dataset_dir, f"{split}_labels.npy")),
        (os.path.join(dataset_dir, f"{split}.npz"),),
        (os.path.join(dataset_dir, f"{split}_data.npz"),),
    ]
    for cand in candidates:
        if len(cand) == 2:
            wfile, lfile = cand
            if os.path.exists(wfile) and os.path.exists(lfile):
                windows = np.load(wfile)
                labels = np.load(lfile)
                return windows, labels
        else:
            path = cand[0]
            if os.path.exists(path):
                windows, labels = _load_np_file(path)
                return windows, labels

    # fallback: attempt to find any *_windows.npy in dir
    for fname in os.listdir(dataset_dir):
        if fname.endswith("_windows.npy") and fname.startswith(split):
            wfile = os.path.join(dataset_dir, fname)
            lfile = os.path.join(dataset_dir, fname.replace("_windows.npy", "_labels.npy"))
            if os.path.exists(lfile):
                return np.load(wfile), np.load(lfile)

    raise FileNotFoundError(f"Could not find dataset files for split '{split}' in {dataset_dir}")

def make_dataloader(dataset_dir: str, split: str, batch_size: int = 64, shuffle: bool = True, num_workers: int = 2) -> DataLoader:
    windows, labels = load_split(dataset_dir, split)
    ds = SequenceDataset(windows, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
