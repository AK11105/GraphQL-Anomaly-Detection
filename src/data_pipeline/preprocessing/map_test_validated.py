#!/usr/bin/env python3
"""
Map test.jsonl IDs back to validated_normal + validated_malicious.

Purpose:
- Recover schema_used and full validated metadata for test split
- Ensure zero leakage (ID-based mapping only)

Input:
  --test test.jsonl
  --validated-normal validated_normal.jsonl
  --validated-malicious validated_malicious.jsonl
  --out mapped_test_validated.jsonl

Output:
  JSONL with records taken directly from validated_* files,
  restricted to IDs present in test.jsonl
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_id(rec: Dict) -> str:
    return (
        rec.get("id")
        or rec.get("query_id")
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Map test IDs to validated records"
    )
    parser.add_argument("--test", required=True, help="test.jsonl (IDs only)")
    parser.add_argument("--validated-normal", required=True)
    parser.add_argument("--validated-malicious", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load test IDs
    # -----------------------------------------------------
    test_ids = set()
    for rec in load_jsonl(args.test):
        rid = extract_id(rec)
        if rid:
            test_ids.add(rid)

    if not test_ids:
        raise RuntimeError("No IDs found in test.jsonl")

    print(f"[INFO] Loaded {len(test_ids)} test IDs")

    # -----------------------------------------------------
    # Build lookup from validated files
    # -----------------------------------------------------
    validated_map: Dict[str, Dict] = {}

    for path in [args.validated_normal, args.validated_malicious]:
        for rec in load_jsonl(path):
            rid = extract_id(rec)
            if rid:
                validated_map[rid] = rec

    print(f"[INFO] Indexed {len(validated_map)} validated records")

    # -----------------------------------------------------
    # Map test → validated
    # -----------------------------------------------------
    mapped = []
    missing = 0

    for rid in test_ids:
        rec = validated_map.get(rid)
        if rec is None:
            missing += 1
        else:
            mapped.append(rec)

    if missing > 0:
        print(f"[WARN] {missing} test IDs not found in validated data")

    # -----------------------------------------------------
    # Write output
    # -----------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in mapped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote {len(mapped)} mapped records → {out_path}")


if __name__ == "__main__":
    main()
