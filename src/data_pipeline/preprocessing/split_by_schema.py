#!/usr/bin/env python3
"""
Split mapped test set into per-schema JSONL files.

Schema is derived from:
  meta.generator_params.schema_used

Example:
  src\\data_pipeline\\schemas\\introspected\\schema_g2.json → schema_g2
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_schema_id(record):
    try:
        schema_used = (
            record.get("meta", {})
            .get("generator_params", {})
            .get("schema_used")
        )
        if not schema_used:
            return None
        return Path(schema_used).stem  # schema_g1, schema_g2, ...
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Split test set by schema (from meta)")
    parser.add_argument("--input", required=True, help="Mapped test JSONL")
    parser.add_argument("--outdir", required=True, help="Output directory")

    args = parser.parse_args()

    buckets = defaultdict(list)
    skipped = 0

    for rec in load_jsonl(args.input):
        schema_id = extract_schema_id(rec)
        if not schema_id:
            skipped += 1
            continue
        buckets[schema_id].append(rec)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for schema, records in sorted(buckets.items()):
        out_path = outdir / f"{schema}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"{schema}: {len(records)} samples → {out_path}")

    print("\nSplit complete.")
    print(f"Schemas found: {sorted(buckets.keys())}")
    if skipped:
        print(f"Skipped records (no schema): {skipped}")


if __name__ == "__main__":
    main()
