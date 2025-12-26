#!/usr/bin/env python3
"""
C1: Build schema-identification dataset from Phase-1 features.

Schema identity is normalized to basename (e.g. schema_g1).
"""

import json
import argparse
from pathlib import Path

from src.ml.models.feature_schema import get_feature_keys


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def normalize_schema_id(schema_used: str) -> str:
    """
    Convert schema path → stable schema ID.
    Example:
      src\\...\\schema_g3.json → schema_g3
    """
    p = Path(schema_used)
    return p.stem  # filename without .json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True)
    parser.add_argument("--malicious", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    feature_keys = get_feature_keys()

    rows = []
    schema_set = set()

    for path in [args.normal, args.malicious]:
        for rec in load_jsonl(path):
            features = rec.get("features")
            if not isinstance(features, dict):
                continue

            schema_used = (
                rec.get("meta", {})
                .get("generator_params", {})
                .get("schema_used")
            )

            if not schema_used:
                continue

            schema_id = normalize_schema_id(schema_used)

            x = [features.get(k, 0.0) or 0.0 for k in feature_keys]

            rows.append({
                "x": x,
                "schema": schema_id
            })
            schema_set.add(schema_id)

    # Stable label mapping
    schema_to_id = {s: i for i, s in enumerate(sorted(schema_set))}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "x": r["x"],
                "y": schema_to_id[r["schema"]],
                "schema": r["schema"]
            }) + "\n")

    print("Schema mapping:", schema_to_id)
    print("Samples:", len(rows))
    print("Output:", out_path)


if __name__ == "__main__":
    main()
