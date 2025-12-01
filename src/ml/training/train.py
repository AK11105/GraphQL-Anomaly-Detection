# ml/training/train.py
"""
Train script for anomaly detectors.

Saves checkpoints:
  - best by val loss: checkpoints/best_model.pt
  - last epoch: checkpoints/last_model.pt

Usage:
  python -m ml.training.train --dataset-dir path/to/final_train_ready --model lstm --epochs 20 --batch-size 128
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.training.dataloader import make_dataloader
from ml.training.metrics import compute_classification_metrics
from ml.models.lstm_detector import LSTMDetector
from ml.models.transformer_detector import TransformerDetector

# default config
DEFAULT_CFG = {
    "feature_dim":  len([]),  # placeholder; will be inferred if possible
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "epochs": 10,
    "batch_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "ml/checkpoints",
    "model": "lstm",
    "seed": 42,
}

def set_seed(seed: int):
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def infer_feature_dim(dataset_dir: str) -> int:
    # load a small sample
    sample_windows, _ = make_dataloader(dataset_dir, "train", batch_size=1, shuffle=False).__iter__().__next__()
    # sample_windows shape (batch, seq_len, feature_dim)
    return sample_windows.shape[2]

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        bsz = xb.size(0)
        epoch_loss += float(loss.item()) * bsz
        n_samples += bsz
    return epoch_loss / (n_samples + 1e-12)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    y_true = []
    y_pred = []
    n_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            preds = model(xb)
            loss = criterion(preds, yb)
            bsz = xb.size(0)
            epoch_loss += float(loss.item()) * bsz
            n_samples += bsz
            y_true.append(yb.cpu().numpy().reshape(-1))
            y_pred.append(preds.cpu().numpy().reshape(-1))
    return epoch_loss / (n_samples + 1e-12), np.concatenate(y_true), np.concatenate(y_pred)

def build_model_by_name(name: str, cfg: dict):
    if name.lower() == "lstm":
        return LSTMDetector(
            feature_dim=cfg["feature_dim"],
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3)
        )
    elif name.lower() == "transformer":
        return TransformerDetector(
            feature_dim=cfg["feature_dim"],
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError("Unknown model: choose 'lstm' or 'transformer'")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to final_train_ready dataset dir")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CFG["weight_decay"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CFG["checkpoint_dir"])
    parser.add_argument("--device", default=DEFAULT_CFG["device"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on val loss")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # dataloaders
    train_loader = make_dataloader(args.dataset_dir, "train", batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(args.dataset_dir, "val", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # infer feature dim
    sample_batch = next(iter(train_loader))[0]
    seq_len = sample_batch.shape[1]
    feature_dim = sample_batch.shape[2]
    cfg = {
        "feature_dim": feature_dim,
        "hidden_dim": 128,
        "num_layers": 2
    }

    model = build_model_by_name(args.model, cfg)
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    patience = args.patience
    no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(1, args.epochs + 1):
        start_time = datetime.utcnow()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
        val_metrics = compute_classification_metrics(y_true, y_pred, threshold=0.5)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={elapsed:.1f}s metrics={val_metrics}")

        # checkpoint last
        last_path = os.path.join(args.checkpoint_dir, "last_model.pt")
        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
            "history": history,
        }, last_path)

        # early stopping / best save
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "history": history,
            }, best_path)
            print(f"  -> saved best model to {best_path}")
        else:
            no_improve += 1
            print(f"  no improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # final saving of training summary
    with open(os.path.join(args.checkpoint_dir, "training_summary.json"), "w") as fh:
        json.dump({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history
        }, fh, indent=2)

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
