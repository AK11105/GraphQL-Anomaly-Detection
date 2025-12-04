#!/usr/bin/env python3
"""
Sequence Construction (Phase 2 – Step 1)

Reads validated normal + malicious JSONL files and produces model-ready
sequence examples.

Output Format (JSONL):

{
  "id": "...",
  "input_text": "<GRAPHQL>\n{ ... }\n</GRAPHQL>\n<FEATURES>\n depth=3 cost=12 fields=9 ...",
  "target_label": "normal" | "malicious",
  "meta": {...}
}

Path:
  src/data_pipeline/generate_sequences.py
"""

from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List


# ---------------------------------------------------------
# Helper: Compact Feature Line
# ---------------------------------------------------------
def build_feature_line(features: Dict[str, Any]) -> str:
    """
    Convert Phase-1 features into a compact textual form.
    Missing values → 0
    """
    if not isinstance(features, dict):
        return "depth=0 cost=0 fields=0"

    # known stable feature keys
    depth = features.get("query_depth") or 0
    cost = features.get("estimated_cost") or features.get("complexity_score") or 0
    fields = features.get("num_fields") or 0
    fragments = features.get("num_fragments") or 0
    tokens = features.get("num_tokens") or 0
    entropy = features.get("entropy") or 0

    return (
        f"depth={depth} "
        f"cost={cost} "
        f"fields={fields} "
        f"fragments={fragments} "
        f"tokens={tokens} "
        f"entropy={entropy}"
    )


# ---------------------------------------------------------
# Helper: Build Input Text Sequence
# ---------------------------------------------------------
def build_input_text(query: str, features: Dict[str, Any]) -> str:
    """
    Final training representation fed to the model.
    """
    feature_line = build_feature_line(features)
    return f"<GRAPHQL>\n{query}\n</GRAPHQL>\n<FEATURES>\n{feature_line}"


# ---------------------------------------------------------
# Process a single validated record
# ---------------------------------------------------------
def convert_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    rid = rec.get("query_id") or str(uuid.uuid4())
    query = rec.get("query", "").strip()
    features = rec.get("features", {})
    label = rec.get("type", "unknown")

    input_text = build_input_text(query, features)

    return {
        "id": rid,
        "input_text": input_text,
        "target_label": label,
        "meta": rec.get("meta", {})
    }


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
                obj = json.loads(line)
                items.append(obj)
            except Exception:
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
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build model-ready sequences from validated GraphQL data")
    parser.add_argument("--normal", required=True, help="Validated normal JSONL")
    parser.add_argument("--malicious", required=True, help="Validated malicious JSONL")
    parser.add_argument("--out", required=True, help="Output sequences.jsonl")

    args = parser.parse_args()

    normal = load_jsonl(args.normal)
    malicious = load_jsonl(args.malicious)

    sequences = []
    for rec in normal:
        sequences.append(convert_record(rec))
    for rec in malicious:
        sequences.append(convert_record(rec))

    # deterministic ordering by id (optional but helpful)
    sequences.sort(key=lambda x: x["id"])

    write_jsonl(args.out, sequences)

    print(f"Sequences built: {len(sequences)} → {args.out}")


if __name__ == "__main__":
    main()
