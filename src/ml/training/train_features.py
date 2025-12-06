import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from src.ml.models.feature_resmlp import FeatureResMLP
from src.ml.training.dataloaders import build_feature_dataloaders


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
    train_loader, val_loader = build_feature_dataloaders(
        TRAIN_PATH, VAL_PATH, FEATURE_KEYS
    )

    model = FeatureResMLP(input_dim=len(FEATURE_KEYS)).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_auc = 0.0

    for epoch in range(1, 31):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.float().to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- VALIDATION --------
        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.float().to(DEVICE)

                logits = model(x)
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

        print(f"[FEATURE] Epoch {epoch} | {metrics} | Saved={saved}")


if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    train()
