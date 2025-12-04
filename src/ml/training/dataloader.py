# src/ml/dataloaders/graphql_dataloaders.py

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# Helper: Load JSONL
# ------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


# ------------------------------------------------------------
# Sequence Dataset (Text + Label)
# ------------------------------------------------------------
class SequenceDataset(Dataset):
    """
    Handles input_text + target_label pairs for transformer models.
    """
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.items = load_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        label_set = {"normal": 0, "malicious": 1}
        for it in self.items:
            it["label_id"] = label_set.get(it["target_label"], -1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        text = rec["input_text"]
        label = rec["label_id"]

        tok = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
            "label": torch.tensor(label, dtype=torch.long),
            "meta": rec.get("meta", {}),
            "id": rec.get("id", None)
        }


# ------------------------------------------------------------
# Collate function for variable-length batches
# ------------------------------------------------------------
def sequence_collate(batch):
    """
    Pads sequences within a batch to uniform size.
    """
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])

    # pad
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "meta": [b["meta"] for b in batch],
        "ids": [b["id"] for b in batch],
    }


# ------------------------------------------------------------
# Feature Dataset (Vector + Label)
# ------------------------------------------------------------
class FeatureDataset(Dataset):
    """
    Uses consolidated numeric feature vectors, not text.
    """
    def __init__(self, jsonl_path: str, feature_keys: List[str]):
        self.items = load_jsonl(jsonl_path)
        self.feature_keys = feature_keys

        label_set = {"normal": 0, "malicious": 1}
        for it in self.items:
            it["label_id"] = label_set.get(it["label"], -1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        vec = torch.tensor([rec.get(k, 0) for k in self.feature_keys], dtype=torch.float)
        label = torch.tensor(rec["label_id"], dtype=torch.long)

        return {
            "features": vec,
            "label": label,
            "id": rec.get("id", None),
            "meta": rec.get("meta", {})
        }


# ------------------------------------------------------------
# Build Dataloaders (Sequence)
# ------------------------------------------------------------
def create_sequence_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 16,
    num_workers: int = 0,
):
    train_ds = SequenceDataset(train_path, tokenizer, max_length)
    val_ds = SequenceDataset(val_path, tokenizer, max_length)
    test_ds = SequenceDataset(test_path, tokenizer, max_length)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, collate_fn=sequence_collate),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=sequence_collate),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=sequence_collate),
    )


# ------------------------------------------------------------
# Build Dataloaders (Feature Vectors)
# ------------------------------------------------------------
def create_feature_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    feature_keys: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_ds = FeatureDataset(train_path, feature_keys)
    val_ds = FeatureDataset(val_path, feature_keys)
    test_ds = FeatureDataset(test_path, feature_keys)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
