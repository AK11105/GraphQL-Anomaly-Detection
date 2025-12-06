import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .feature_datasets import FeatureDataset
from .text_datasets import TextDataset
LABEL_MAP = {
    "normal": 0,
    "malicious": 1,
}


def build_feature_dataloaders(
    train_path: str,
    val_path: str,
    feature_keys: list,
    batch_size_train: int = 1024,
    batch_size_val: int = 2048,
    num_workers: int = 2,
):
    train_ds = FeatureDataset(train_path, feature_keys)
    val_ds = FeatureDataset(val_path, feature_keys)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_text_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    max_len: int = 512,
    batch_size_train: int = 32,
    batch_size_val: int = 64,
    num_workers: int = 2,
):
    train_ds = TextDataset(train_path, tokenizer, max_len)
    val_ds = TextDataset(val_path, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
