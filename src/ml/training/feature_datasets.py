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
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)