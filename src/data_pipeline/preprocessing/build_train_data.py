#!/usr/bin/env python3
"""
Unified Training Dataset Builder (Phase 2 – Step 3)

Rules:
- Include:
      validated_normal        → label: normal
      validated_malicious     → label: malicious
      invalid_malicious       → label: malicious

- Exclude:
      invalid_normal          (generator mistakes)

Output:
  dataset/unified/all.jsonl
  dataset/unified/stats.json
"""

from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List


# ------------------------------
# Load JSONL helper
# ------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


# ------------------------------
# Convert record → unified row
# ------------------------------
def canonicalize(rec: Dict[str, Any], forced_label: str | None = None) -> Dict[str, Any]:
    rid = rec.get("query_id") or str(uuid.uuid4())
    lbl = forced_label or rec.get("type") or "unknown"

    return {
        "id": rid,
        "label": lbl,
        "query": rec.get("query", ""),
        "features": rec.get("features", {}),
        "meta": rec.get("meta", {}),
    }


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build unified training dataset")
    parser.add_argument("--validated-normal", required=True)
    parser.add_argument("--validated-malicious", required=True)
    parser.add_argument("--invalid-malicious", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    # Load data
    normal = load_jsonl(args.validated_normal)
    mal_valid = load_jsonl(args.validated_malicious)
    mal_invalid = load_jsonl(args.invalid_malicious)

    unified = []

    # Normal → include
    for r in normal:
        unified.append(canonicalize(r, forced_label="normal"))

    # Malicious validated → include
    for r in mal_valid:
        unified.append(canonicalize(r, forced_label="malicious"))

    # Malicious invalid → include
    for r in mal_invalid:
        unified.append(canonicalize(r, forced_label="malicious"))

    # Sort for determinism
    unified.sort(key=lambda x: x["id"])

    out_dir = Path(args.out)
    out_path = out_dir / "all.jsonl"
    stats_path = out_dir / "stats.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write unified file
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in unified:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write stats
    stats = {
        "total": len(unified),
        "normal": len(normal),
        "malicious_valid": len(mal_valid),
        "malicious_invalid": len(mal_invalid),
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print("Unified dataset built:")
    print(" total:", len(unified))
    print(" normal:", len(normal))
    print(" malicious_valid:", len(mal_valid))
    print(" malicious_invalid:", len(mal_invalid))
    print("→", out_path)
    print("→", stats_path)


if __name__ == "__main__":
    main()
