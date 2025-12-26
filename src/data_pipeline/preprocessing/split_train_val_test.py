#!/usr/bin/env python3
"""
Dataset Splitter (Phase 2)

Takes unified sequences.jsonl and produces train/val/test splits.

Features:
 - Stratified split on target_label ("normal" / "malicious")
 - Deterministic shuffling via --seed
 - Configurable split ratios
 - Writes JSONL files in:
       dataset/train/
       dataset/val/
       dataset/test/

Input Format (from generate_sequences):
{
  "id": "...",
  "input_text": "...",
  "target_label": "normal" | "malicious",
  "meta": {...}
}
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except:
                pass
    return items


# ---------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------
def write_jsonl(path: str, items: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# Stratified split
# ---------------------------------------------------------
def stratified_split(
    items: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)

    buckets = {}
    for rec in items:
        label = rec.get("target_label", "unknown")
        buckets.setdefault(label, []).append(rec)

    train_set, val_set, test_set = [], [], []

    for label, group in buckets.items():
        random.shuffle(group)

        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_set.extend(group[:n_train])
        val_set.extend(group[n_train:n_train + n_val])
        test_set.extend(group[n_train + n_val:])

    # final shuffle (optional)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stratified dataset split for GraphQL anomaly detection")
    parser.add_argument("--input", required=True, help="Unified sequences.jsonl file")
    parser.add_argument("--outdir", required=True, help="Output dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data = load_jsonl(args.input)

    train, val, test = stratified_split(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    write_jsonl(Path(args.outdir) / "train/train.jsonl", train)
    write_jsonl(Path(args.outdir) / "val/val.jsonl", val)
    write_jsonl(Path(args.outdir) / "test/test.jsonl", test)

    print(f"Split complete:")
    print(f"  train = {len(train)}")
    print(f"  val   = {len(val)}")
    print(f"  test  = {len(test)}")
    print(f"Output → {args.outdir}")


if __name__ == "__main__":
    main()
