#!/usr/bin/env python3
"""
Split Feature Matrix (Phase 2)

Input:
  A consolidated feature matrix JSONL file with records of the form:
  {
    "id": "...",
    "features": [...],
    "target_label": "normal" | "malicious",
    "meta": {...}
  }

Output:
  Stratified splits:
    dataset/features/train.jsonl
    dataset/features/val.jsonl
    dataset/features/test.jsonl

Usage:
  python src/data_pipeline/split_features.py \
      --input src/data_pipeline/dataset/features/feature_matrix.jsonl \
      --outdir src/data_pipeline/dataset/features \
      --train-ratio 0.8 \
      --val-ratio 0.1 \
      --test-ratio 0.1 \
      --seed 42
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------
def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


# ---------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------
def write_jsonl(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# Stratified Split
# ---------------------------------------------------------
def stratified_split(items: List[Dict], train_r: float, val_r: float, test_r: float, seed: int):
    random.seed(seed)

    # group by label
    buckets: Dict[str, List[Dict]] = {}
    for it in items:
        label = it.get("target_label", "unknown")
        buckets.setdefault(label, []).append(it)

    train, val, test = [], [], []

    for label, bucket in buckets.items():
        random.shuffle(bucket)

        n = len(bucket)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    return train, val, test


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stratified split for feature matrix")
    parser.add_argument("--input", required=True, help="Input feature_matrix.jsonl")
    parser.add_argument("--outdir", required=True, help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load feature matrix
    items = load_jsonl(args.input)

    # Perform split
    train, val, test = stratified_split(
        items,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    outdir = Path(args.outdir)

    write_jsonl(outdir / "train.jsonl", train)
    write_jsonl(outdir / "val.jsonl", val)
    write_jsonl(outdir / "test.jsonl", test)

    print(f"Feature split complete:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    print(f"→ Written to {outdir}")


if __name__ == "__main__":
    main()
