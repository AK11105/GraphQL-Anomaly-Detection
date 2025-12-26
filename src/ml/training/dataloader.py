import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .dataset import FeatureDataset
from .dataset import TextDataset
from .dataset import FeatureSchemaDataset
from .dataset import SchemaTextDataset
from transformers import AutoTokenizer

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


def build_feature_schema_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 128,
    num_workers: int = 0,
):
    train_ds = FeatureSchemaDataset(train_path)
    val_ds   = FeatureSchemaDataset(val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_schema_text_dataloaders(
    train_path,
    val_path,
    model_name="roberta-base",
    batch_size=16,
    max_len=256,
    num_workers=0,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = SchemaTextDataset(train_path, tokenizer, max_len)
    val_ds   = SchemaTextDataset(val_path, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer