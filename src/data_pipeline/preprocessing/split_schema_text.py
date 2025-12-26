#!/usr/bin/env python3
"""
Train–Val Dataset Splitter (Phase 2 / C studies)

- Stratified split on target_label
- Deterministic via seed
- NO test split
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def stratified_train_val_split(
    items: List[Dict[str, Any]],
    train_ratio: float,
    seed: int,
):
    random.seed(seed)

    buckets = {}
    for rec in items:
        label = rec.get("target_label", "unknown")
        buckets.setdefault(label, []).append(rec)

    train, val = [], []

    for label, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:])

    random.shuffle(train)
    random.shuffle(val)

    return train, val


def main():
    parser = argparse.ArgumentParser(description="Stratified train–val split")
    parser.add_argument("--input", required=True, help="sequences.jsonl")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data = load_jsonl(args.input)

    train, val = stratified_train_val_split(
        data,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    outdir = Path(args.outdir)
    write_jsonl(outdir / "train.jsonl", train)
    write_jsonl(outdir / "val.jsonl", val)

    print("Split complete:")
    print(f"  train = {len(train)}")
    print(f"  val   = {len(val)}")
    print(f"Output → {outdir}")


if __name__ == "__main__":
    main()
