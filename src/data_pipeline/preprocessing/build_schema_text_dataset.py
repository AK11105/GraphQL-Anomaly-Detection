#!/usr/bin/env python3
"""
C2′: Build TEXT-ONLY schema identification dataset.

Uses:
  - validated_normal.jsonl
  - validated_malicious.jsonl
  - invalid_malicious.jsonl

Task:
  Predict schema_id from raw GraphQL query text ONLY.

Output JSONL:
{
  "text": "...",
  "y": int,
  "schema": "schema_g1"
}
"""

import json
import argparse
from pathlib import Path


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def normalize_schema_id(schema_used: str) -> str:
    """
    Convert schema path → stable schema ID.
    Example:
      src/.../schema_g3.json → schema_g3
    """
    return Path(schema_used).stem


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build text-only schema ID dataset (C2′)")
    parser.add_argument("--validated-normal", required=True)
    parser.add_argument("--validated-malicious", required=True)
    parser.add_argument("--invalid-malicious", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    records = []
    schema_set = set()

    input_files = [
        args.validated_normal,
        args.validated_malicious,
        args.invalid_malicious,
    ]

    for path in input_files:
        for rec in load_jsonl(path):
            query = rec.get("query")
            if not isinstance(query, str) or not query.strip():
                continue

            schema_used = (
                rec.get("meta", {})
                .get("generator_params", {})
                .get("schema_used")
            )
            if not schema_used:
                continue

            schema_id = normalize_schema_id(schema_used)
            schema_set.add(schema_id)

            records.append({
                "text": query.strip(),
                "schema": schema_id,
            })

    # Stable label mapping
    schema_to_id = {s: i for i, s in enumerate(sorted(schema_set))}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "text": r["text"],
                "y": schema_to_id[r["schema"]],
                "schema": r["schema"],
            }, ensure_ascii=False) + "\n")

    print("Schema mapping:", schema_to_id)
    print("Samples:", len(records))
    print("Output:", out_path)


if __name__ == "__main__":
    main()
