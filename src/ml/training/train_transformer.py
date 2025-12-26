# src/ml/training/train_sota_transformer.py

import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from src.ml.models.sota_transformer import SOTATransformerClassifier
from src.ml.training.dataloader import build_text_dataloaders
from src.ml.training.metrics import compute_classification_metrics


# =========================
# Reproducibility
# =========================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "roberta-base"

SEQ_TRAIN_PATH = "dataset/sequences/train.jsonl"
SEQ_VAL_PATH   = "dataset/sequences/val.jsonl"

ARTIFACT_PATH = Path("src/ml/artifacts/transformer_best.pt")
METRICS_PATH  = Path("src/ml/artifacts/transformer_metrics.json")

EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

MAX_LEN = 256
BATCH_SIZE_TRAIN = 16    # safer for transformers on Kaggle
BATCH_SIZE_VAL   = 32


# =========================
# Training Loop
# =========================
def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader, val_loader = build_text_dataloaders(
        SEQ_TRAIN_PATH,
        SEQ_VAL_PATH,
        tokenizer,
        max_len=MAX_LEN,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
    )

    model = SOTATransformerClassifier(
        model_name=MODEL_NAME,
        dropout=0.2,
        freeze_encoder=False,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_auc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        # =========================
        # Train
        # =========================
        model.train()
        epoch_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"[TRANSFORMER][Epoch {epoch:02d}] Train",
            leave=False,
        )

        for batch in train_pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # =========================
        # Validation
        # =========================
        model.eval()
        all_logits, all_labels = [], []

        val_pbar = tqdm(
            val_loader,
            desc=f"[TRANSFORMER][Epoch {epoch:02d}] Val",
            leave=False,
        )

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                logits = model(input_ids, attention_mask)
                all_logits.append(logits)
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = labels.cpu().numpy()

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred_proba=probs,
            threshold=0.5,
        )

        metrics.update({
            "epoch": epoch,
            "train_loss": avg_train_loss,
        })

        history.append(metrics)

        # =========================
        # Checkpoint
        # =========================
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ARTIFACT_PATH)
            saved = True
        else:
            saved = False

        print(
            f"[TRANSFORMER] Epoch {epoch:02d} | "
            f"Loss={avg_train_loss:.4f} | "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"FPR={metrics['fpr']:.4f} | "
            f"Saved={saved}"
        )

    # =========================
    # Save Metrics
    # =========================
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best AUC: {best_auc:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    train()
