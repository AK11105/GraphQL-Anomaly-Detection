# src/ml/training/train_schema_text.py

import json
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from src.ml.training.dataloader import build_schema_text_dataloaders
from src.ml.models.text_schema_classifier import SchemaTextClassifier


def train_schema_text():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_SCHEMAS = 3
    MODEL_NAME = "roberta-base"
    EPOCHS = 10
    LR = 2e-5

    train_loader, val_loader, _ = build_schema_text_dataloaders(
        train_path="dataset/schema_text/train.jsonl",
        val_path="dataset/schema_text/val.jsonl",
        model_name=MODEL_NAME,
        batch_size=16,
    )

    model = SchemaTextClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_SCHEMAS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0

        for ids, mask, y in tqdm(train_loader, desc=f"Epoch {epoch:02d} Train"):
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)

            logits = model(ids, mask)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ---- VALIDATE ----
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for ids, mask, y in tqdm(val_loader, desc=f"Epoch {epoch:02d} Val"):
                ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                preds = model(ids, mask).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_acc": val_acc,
        })

        print(f"[Epoch {epoch:02d}] val_acc={val_acc:.4f}")

    # ---- SAVE ----
    out = Path("analysis")
    out.mkdir(exist_ok=True)

    with open(out / "c2_text_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("C2′ training complete → analysis/c2_text_history.json")
