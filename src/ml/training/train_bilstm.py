import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.ml.models.attentive_bilstm import AttentiveBiLSTM
from src.ml.training.dataloader import build_text_dataloaders
from src.ml.training.metrics import compute_classification_metrics


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "roberta-base"

SEQ_TRAIN_PATH = "dataset/sequences/train.jsonl"
SEQ_VAL_PATH   = "dataset/sequences/val.jsonl"

ARTIFACT_PATH = Path("src/ml/artifacts/bilstm_best.pt")
METRICS_PATH  = Path("src/ml/artifacts/bilstm_metrics.json")

EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0


# =========================
# Training Loop
# =========================
def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad_idx = tokenizer.pad_token_id

    train_loader, val_loader = build_text_dataloaders(
        SEQ_TRAIN_PATH,
        SEQ_VAL_PATH,
        tokenizer,
        max_len=256,              # ✅ reduced (safe + faster)
        batch_size_train=32,
        batch_size_val=64,
    )

    model = AttentiveBiLSTM(
        vocab_size=tokenizer.vocab_size,
        pad_idx=pad_idx,
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
            desc=f"[BiLSTM][Epoch {epoch:02d}] Train",
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
            desc=f"[BiLSTM][Epoch {epoch:02d}] Val",
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
            torch.save(model.state_dict(), ARTIFACT_PATH)
            saved = True
        else:
            saved = False

        print(
            f"[BiLSTM] Epoch {epoch:02d} | "
            f"Loss={avg_train_loss:.4f} | "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"FPR={metrics['fpr']:.4f} | "
            f"Saved={saved}"
        )

    # =========================
    # Save metrics JSON
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
