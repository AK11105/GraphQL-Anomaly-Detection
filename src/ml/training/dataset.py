# src/ml/training/datasets.py  (or directly in your notebook)

import json
import torch
from torch.utils.data import Dataset
import numpy as np



LABEL_MAP = {
    "normal": 0,
    "malicious": 1,
}


class FeatureDataset(Dataset):
    """
    Expected JSONL format per row:
    {
        "id": "...",
        "label": "normal" | "malicious",
        "features": { ... }
    }
    """

    def __init__(self, jsonl_path: str, feature_keys: list):
        self.feature_keys = feature_keys
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                features = obj["features"]
                label = obj["label"]

                x = np.array(
                    [float(features[k]) for k in feature_keys],
                    dtype=np.float32
                )

                y = LABEL_MAP[label]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    

import json
import torch
from torch.utils.data import Dataset


LABEL_MAP = {
    "normal": 0,
    "malicious": 1,
}


# ============================================================
# =============== TEXT DATASET (SEQUENCES) ===================
# ============================================================

class TextDataset(Dataset):
    """
    Expected JSONL format per row:
    {
        "id": "...",
        "input_text": "...",
        "target_label": "normal" | "malicious",
        "meta": {...}
    }
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                text = obj["input_text"]
                label = obj["target_label"]

                y = LABEL_MAP[label]
                self.samples.append((text, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, y = self.samples[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.float32),
        }

# ============================================================
# ======= FEATURE SCHEMA DATASET (SCHEMA AGNOSTICITY) ========
# ============================================================


class FeatureSchemaDataset(Dataset):
    """
    Dataset for schema-identification using Phase-1 feature vectors.

    Each JSONL line:
      {
        "x": [...],          # feature vector
        "y": int,            # schema label
        "schema": "schema_g1"
      }
    """

    def __init__(self, jsonl_path: str):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        x = torch.tensor(row["x"], dtype=torch.float32)
        y = torch.tensor(row["y"], dtype=torch.long)
        return x, y

class SchemaTextDataset(Dataset):
    """
    JSONL format:
    {
      "text": "...GraphQL query...",
      "y": int,
      "schema": "schema_g1"
    }
    """

    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(row["y"], dtype=torch.long),
        )
