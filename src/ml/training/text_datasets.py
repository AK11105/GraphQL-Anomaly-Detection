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
            "labels": torch.tensor(y, dtype=torch.long),
        }