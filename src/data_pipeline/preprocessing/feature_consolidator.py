#!/usr/bin/env python3
"""
Feature Consolidation (Phase 2 – Step 2)

Reads validated normal + malicious JSONL files and produces:
 - feature_matrix.jsonl   (full ordered feature rows)
 - feature_matrix.parquet (optional, if pyarrow available)
 - manifest.json          (summary + feature schema)

Each output row:
{
  "id": "...",
  "label": "normal" | "malicious",
  "features": {
      "<feature_key>": value,
      ...
  }
}
"""

from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List

from src.ml.models.feature_schema import get_feature_keys


# ---------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


# ---------------------------------------------------------
# Build ordered feature row
# ---------------------------------------------------------
def extract_feature_row(rec: Dict[str, Any], feature_keys: List[str]) -> Dict[str, Any]:
    features = rec.get("features", {})
    row = {}

    # ensure all features exist (missing → 0)
    for k in feature_keys:
        v = features.get(k)
        if v is None:
            v = 0
        row[k] = v

    return {
        "id": rec.get("query_id") or str(uuid.uuid4()),
        "label": rec.get("type", "unknown"),
        "features": row,
    }


# ---------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------
def write_jsonl(path: Path, items: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# Optional Parquet Output
# ---------------------------------------------------------
def write_parquet(path: Path, rows: List[Dict[str, Any]], feature_keys: List[str]):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        print("pyarrow not installed → skipping .parquet output")
        return

    ids = [r["id"] for r in rows]
    labels = [r["label"] for r in rows]

    cols = {k: [r["features"][k] for r in rows] for k in feature_keys}
    cols["id"] = ids
    cols["label"] = labels

    table = pa.table(cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"Wrote Parquet → {path}")


# ---------------------------------------------------------
# Manifest
# ---------------------------------------------------------
def write_manifest(path: Path, num_rows: int, feature_keys: List[str]):
    manifest = {
        "rows": num_rows,
        "features": feature_keys,
        "version": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Consolidate features into a unified ML matrix.")
    parser.add_argument("--normal", required=True, help="Validated normal JSONL")
    parser.add_argument("--malicious", required=True, help="Validated malicious JSONL")
    parser.add_argument("--out", required=True, help="Output directory for feature matrix")

    args = parser.parse_args()

    feature_keys = get_feature_keys()

    normal = load_jsonl(args.normal)
    malicious = load_jsonl(args.malicious)

    rows = []
    for rec in normal:
        rows.append(extract_feature_row(rec, feature_keys))
    for rec in malicious:
        rows.append(extract_feature_row(rec, feature_keys))

    rows.sort(key=lambda x: x["id"])

    outdir = Path(args.out)
    jsonl_path = outdir / "feature_matrix.jsonl"
    parquet_path = outdir / "feature_matrix.parquet"
    manifest_path = outdir / "manifest.json"

    write_jsonl(jsonl_path, rows)
    write_parquet(parquet_path, rows, feature_keys)
    write_manifest(manifest_path, len(rows), feature_keys)

    print(f"Feature matrix built: {len(rows)} rows")
    print(f"→ {jsonl_path}")
    print(f"→ {manifest_path}")


if __name__ == "__main__":
    main()
