import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from src.ml.models.attentive_bilstm import AttentiveBiLSTM
from src.ml.training.dataloaders import build_text_dataloaders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACT_PATH = "src/ml/artifacts/bilstm_best.pt"

SEQ_TRAIN_PATH = "dataset/sequences/train.jsonl"
SEQ_VAL_PATH   = "dataset/sequences/val.jsonl"

MODEL_NAME = "roberta-base"


def compute_metrics(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    auc = roc_auc_score(y, probs)
    preds = (probs >= 0.5).astype(int)

    p, r, f, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )

    return {"auc": float(auc), "precision": float(p), "recall": float(r), "f1": float(f)}


def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader, val_loader = build_text_dataloaders(
        SEQ_TRAIN_PATH, SEQ_VAL_PATH, tokenizer
    )

    model = AttentiveBiLSTM(vocab_size=tokenizer.vocab_size).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    best_auc = 0.0

    for epoch in range(1, 21):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].float().to(DEVICE)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # -------- VALIDATION --------
        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                y = batch["labels"].float().to(DEVICE)

                logits = model(input_ids, attn_mask)
                all_logits.append(logits)
                all_labels.append(y)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        metrics = compute_metrics(logits, labels)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), ARTIFACT_PATH)
            saved = True
        else:
            saved = False

        print(f"[BiLSTM] Epoch {epoch} | {metrics} | Saved={saved}")


if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    train()
