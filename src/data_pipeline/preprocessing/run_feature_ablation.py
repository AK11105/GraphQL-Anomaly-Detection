#!/usr/bin/env python3
"""
Run systematic ablation studies for schema identifiability (C1).

Each ablation specifies a subset of FEATURE_KEYS to keep.
Trains the same model + hyperparams for fair comparison.
Logs results to JSON for paper-ready tables.
"""

import json
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm

from src.ml.models.feature_schema import FEATURE_KEYS
from src.ml.models.feature_schema_classifier import FeatureSchemaClassifier
from src.ml.training.dataloader import build_feature_schema_dataloaders


# ---------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------
ABLATIONS = {
    # ---------- Baseline ----------
    "B0_all": FEATURE_KEYS,

    # ---------- Remove groups ----------
    "A1_no_text": [
        k for k in FEATURE_KEYS
        if k not in {"entropy", "query_length", "num_tokens"}
    ],

    "A2_no_cost": [
        k for k in FEATURE_KEYS
        if k not in {"estimated_cost", "complexity_score"}
    ],

    "A3_no_depth": [
        k for k in FEATURE_KEYS
        if k not in {
            "query_depth",
            "avg_depth",
            "branching_factor",
            "node_count",
            "num_nested_selections",
        }
    ],

    "A4_no_counts": [
        k for k in FEATURE_KEYS
        if k not in {
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
        }
    ],

    # ---------- Single-family ----------
    "B1_depth_only": [
        "query_depth",
        "avg_depth",
        "branching_factor",
        "node_count",
        "num_nested_selections",
    ],

    "B2_counts_only": [
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
    ],

    "B3_cost_only": [
        "estimated_cost",
        "complexity_score",
    ],

    "B4_text_only": [
        "entropy",
        "query_length",
        "num_tokens",
    ],

    # ---------- Minimal ----------
    "C1_core_structural": [
        "query_depth",
        "branching_factor",
        "num_fields",
    ],
}


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def filter_features(x, keys, key_to_idx):
    idxs = [key_to_idx[k] for k in keys]
    return x[:, idxs]


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_and_eval(
    train_loader,
    val_loader,
    feature_keys,
    device,
    epochs=20,
):
    key_to_idx = {k: i for i, k in enumerate(FEATURE_KEYS)}

    model = FeatureSchemaClassifier(
        input_dim=len(feature_keys),
        num_classes=3,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = filter_features(x, feature_keys, key_to_idx)

            loss = criterion(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = filter_features(x, feature_keys, key_to_idx)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        best_val = max(best_val, acc)

    return best_val


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", default="analysis/ablation_results.json")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_feature_schema_dataloaders(
        train_path=args.train,
        val_path=args.val,
        batch_size=128,
    )

    results = {}

    for name, keys in tqdm(ABLATIONS.items(), desc="Ablations"):
        acc = train_and_eval(
            train_loader,
            val_loader,
            keys,
            device,
            epochs=args.epochs,
        )

        results[name] = {
            "num_features": len(keys),
            "val_accuracy": acc,
            "features": keys,
        }

        print(f"{name:18s} | acc={acc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\nAblation results written to:", out_path)


if __name__ == "__main__":
    main()
