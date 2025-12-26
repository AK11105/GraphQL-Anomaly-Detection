# src/ml/training/train_feature_resmlp.py

import torch
import torch.nn as nn
import numpy as np

from src.ml.models.feature_resmlp import FeatureResMLP
from src.ml.training.dataloader import build_feature_dataloaders
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
ARTIFACT_PATH = "src/ml/artifacts/feature_resmlp_best.pt"

FEATURE_KEYS = [
    "num_fields",
    "num_fragments",
    "num_directives",
    "num_aliases",
    "num_operations",
    "num_mutations",
    "num_subscriptions",
    "num_variables",
    "num_arguments",
    "num_introspection_ops",
    "query_depth",
    "avg_depth",
    "branching_factor",
    "node_count",
    "num_nested_selections",
    "estimated_cost",
    "complexity_score",
    "entropy",
    "query_length",
    "num_tokens",
    "has_error",
]

TRAIN_PATH = "dataset/features/train.jsonl"
VAL_PATH   = "dataset/features/val.jsonl"


# =========================
# Training Loop
# =========================
def train():
    train_loader, val_loader = build_feature_dataloaders(
        TRAIN_PATH,
        VAL_PATH,
        FEATURE_KEYS,
    )

    model = FeatureResMLP(input_dim=len(FEATURE_KEYS)).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
    )

    best_auc = 0.0

    for epoch in range(1, 31):
        # -------- TRAIN --------
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x).squeeze(-1)
                all_logits.append(logits)
                all_labels.append(y)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        # ---- metrics via shared utility ----
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = labels.cpu().numpy()

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred_proba=probs,
            threshold=0.5,
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            torch.save(model.state_dict(), ARTIFACT_PATH)
            saved = True
        else:
            saved = False

        print(
            f"[FEATURE] Epoch {epoch:02d} | "
            f"Loss={avg_loss:.4f} | "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"FPR={metrics['fpr']:.4f} | "
            f"Saved={saved}"
        )


# =========================
# Entry
# =========================
if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    train()
