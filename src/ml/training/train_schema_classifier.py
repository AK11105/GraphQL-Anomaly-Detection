import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

from src.ml.training.dataloader import build_feature_schema_dataloaders
from src.ml.models.feature_schema_classifier import FeatureSchemaClassifier
from src.ml.models.feature_schema import get_feature_keys


def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    FEATURE_DIM = len(get_feature_keys())
    NUM_SCHEMAS = 3   # schema_g1, schema_g2, schema_g3
    EPOCHS = 20
    BATCH_SIZE = 128
    LR = 1e-3

    # -----------------------------
    # Data
    # -----------------------------
    train_loader, val_loader = build_feature_schema_dataloaders(
        train_path="dataset/schema_id/train.jsonl",
        val_path="dataset/schema_id/val.jsonl",
        batch_size=BATCH_SIZE,
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = FeatureSchemaClassifier(
        input_dim=FEATURE_DIM,
        num_classes=NUM_SCHEMAS,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -----------------------------
    # Logging containers
    # -----------------------------
    history = []
    best_val_acc = 0.0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, EPOCHS + 1):
        # ===== TRAIN =====
        model.train()
        total_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"[Epoch {epoch:02d}] Train",
            leave=False,
        )

        for x, y in train_pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        correct = 0
        total = 0

        # for deeper analysis
        confusion = defaultdict(Counter)

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)
                preds = logits.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                for t, p in zip(y.tolist(), preds.tolist()):
                    confusion[t][p] += 1

        val_acc = correct / max(total, 1)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    # -----------------------------
    # Save probe results
    # -----------------------------
    out_dir = Path("analysis/c1_schema_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    with (out_dir / "confusion.json").open("w") as f:
        json.dump(
            {str(k): dict(v) for k, v in confusion.items()},
            f,
            indent=2
        )

    print("\n=== C1 SUMMARY ===")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    train()
