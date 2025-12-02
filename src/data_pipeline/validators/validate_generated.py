#!/usr/bin/env python3
"""
validate_generated.py (patched for per-record schema)

Validates generated GraphQL JSONL produced by generate_normal.py (or other generators).
Key behavior:
- Loads per-record schema from record.meta.generator_params.schema_used when present
- Caches SDLs (introspection -> SDL) for performance
- Calls Phase-1 extractor: extract_features(query, schema_sdl)
- Writes validated and invalid JSONL and a summary JSON
"""

from __future__ import annotations
import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# graphql-core helpers
from graphql import build_client_schema, print_schema

# ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.ingestion.parser.feature_extractor import extract_features
except Exception as e:
    raise ImportError(
        "Unable to import extract_features from src.ingestion.parser.feature_extractor. "
        "Make sure path is correct and src is a package. Original error: " + repr(e)
    )

# Cache for SDL per introspection path/key
SCHEMA_SDL_CACHE: Dict[str, str] = {}


def load_introspection_as_sdl(introspection_path: str) -> str:
    p = Path(introspection_path)
    if not p.exists():
        raise FileNotFoundError(f"Introspection file not found: {introspection_path}")
    with p.open("r", encoding="utf-8") as fh:
        intros = json.load(fh)
    schema = build_client_schema(intros)
    sdl = print_schema(schema)
    return sdl


def get_schema_sdl_for_record(record: Dict[str, Any], default_schema_sdl: Optional[str]) -> Optional[str]:
    """
    Determine SDL for a record:
      1. If record.meta.generator_params.schema_used exists and points to an introspection file -> load (cached)
      2. Else fall back to default_schema_sdl (CLI --schema-introspection)
      3. Else None
    """
    try:
        meta = record.get("meta", {}) or {}
        gen_params = meta.get("generator_params", {}) or {}
        schema_used = gen_params.get("schema_used")
        if schema_used:
            # return cached if present
            if schema_used in SCHEMA_SDL_CACHE:
                return SCHEMA_SDL_CACHE[schema_used]
            # attempt to open path as given
            p = Path(schema_used)
            if not p.exists():
                # try relative to cwd
                p = (Path.cwd() / schema_used)
            if not p.exists():
                # not found; fall back to default
                return default_schema_sdl
            try:
                sdl = load_introspection_as_sdl(str(p))
                SCHEMA_SDL_CACHE[schema_used] = sdl
                return sdl
            except Exception:
                return default_schema_sdl
    except Exception:
        pass
    return default_schema_sdl


def normalize_input_line(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            q = obj.get("query") or obj.get("graphql") or obj.get("q") or obj.get("query_str")
            if not q:
                return None
            record = {
                "query_id": obj.get("query_id") or obj.get("id") or str(uuid.uuid4()),
                "query": q,
                "type": obj.get("type", "unknown"),
                "meta": obj.get("meta", {}),
            }
            if "generator_params" in obj:
                record.setdefault("meta", {})["generator_params"] = obj["generator_params"]
            return record
        return None
    except json.JSONDecodeError:
        return {"query_id": str(uuid.uuid4()), "query": s, "type": "unknown", "meta": {}}


def validate_record(record: Dict[str, Any], default_schema_sdl: Optional[str]) -> Tuple[bool, Dict[str, Any]]:
    query = record.get("query")
    if not query or not isinstance(query, str):
        out = dict(record)
        out["error"] = "missing_or_invalid_query"
        return False, out

    # determine SDL for this record (prefer per-record schema_used)
    effective_sdl = get_schema_sdl_for_record(record, default_schema_sdl)

    try:
        res = extract_features(query, effective_sdl)
    except Exception as e:
        return False, {**record, "error": f"extractor_call_exception: {repr(e)}"}

    if not isinstance(res, dict):
        return False, {**record, "error": "extractor_returned_non_dict", "raw": str(res)}

    # check extractor-reported errors (ast_error, cost_error)
    if res.get("error") or res.get("ast_error") or res.get("cost_error") or res.get("features") is None:
        err = {}
        if res.get("error"):
            err["error"] = res.get("error")
        if res.get("ast_error"):
            err["ast_error"] = res.get("ast_error")
        if res.get("cost_error"):
            err["cost_error"] = res.get("cost_error")
        output = dict(record)
        output["error"] = err or "unknown_extractor_error"
        output["extractor_raw"] = res
        return False, output

    features = res.get("features", {})
    if isinstance(features, dict) and "ast" in features:
        if isinstance(features["ast"], (dict, list)):
            features["ast"] = "<omitted>"
    out = dict(record)
    out["features"] = features
    out["extractor_meta"] = {"ast_error": res.get("ast_error"), "cost_error": res.get("cost_error")}
    return True, out


def process_input(
    input_path: str,
    out_validated: str,
    out_invalid: str,
    default_schema_introspection: Optional[str] = None,
    workers: int = 4,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    in_p = Path(input_path)
    input_files = []
    if in_p.is_dir():
        for f in sorted(in_p.iterdir()):
            if f.suffix == ".jsonl":
                input_files.append(str(f))
    elif in_p.is_file():
        input_files = [str(in_p)]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    Path(out_validated).parent.mkdir(parents=True, exist_ok=True)
    Path(out_invalid).parent.mkdir(parents=True, exist_ok=True)

    default_schema_sdl: Optional[str] = None
    if default_schema_introspection:
        default_schema_sdl = load_introspection_as_sdl(default_schema_introspection)

    total = 0
    valid_count = 0
    invalid_count = 0
    depth_hist = Counter()
    cost_hist = Counter()
    feature_keys_seen = Counter()
    error_reasons = Counter()

    with open(out_validated, "w", encoding="utf-8") as vf, open(out_invalid, "w", encoding="utf-8") as xf:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for fp in input_files:
                with open(fp, "r", encoding="utf-8") as inf:
                    for line in inf:
                        if max_records is not None and total >= max_records:
                            break
                        total += 1
                        rec = normalize_input_line(line)
                        if rec is None:
                            invalid_count += 1
                            error_reasons["normalize_failed"] += 1
                            xf.write(json.dumps({"raw_line": line.strip(), "error": "normalize_failed"}, ensure_ascii=False) + "\n")
                            continue
                        futures.append(ex.submit(validate_record, rec, default_schema_sdl))
            for fut in as_completed(futures):
                ok, out = fut.result()
                if ok:
                    valid_count += 1
                    feats = out.get("features", {})
                    if isinstance(feats, dict):
                        d = feats.get("query_depth")
                        c = feats.get("estimated_cost") or feats.get("complexity_score")
                        if isinstance(d, (int, float)):
                            depth_hist[int(d)] += 1
                        if isinstance(c, (int, float)):
                            try:
                                b = int(max(0, min(999999, round(c))))
                                cost_hist[b] += 1
                            except Exception:
                                pass
                        for k in feats.keys():
                            feature_keys_seen[k] += 1
                    vf.write(json.dumps(out, ensure_ascii=False) + "\n")
                else:
                    invalid_count += 1
                    err = out.get("error", "unknown_error")
                    if isinstance(err, dict):
                        for k in err.keys():
                            error_reasons[k] += 1
                    else:
                        error_reasons[str(err)] += 1
                    xf.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "total_processed": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "depth_buckets_top": depth_hist.most_common(20),
        "cost_buckets_top": cost_hist.most_common(20),
        "feature_keys_seen_top": feature_keys_seen.most_common(100),
        "error_reasons_top": error_reasons.most_common(20),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate generated GraphQL queries using Phase-1 extractor (per-record schema support).")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file or directory of .jsonl files (raw generated queries).")
    parser.add_argument("--out-validated", "-o", default="src/data_pipeline/dataset/processed/validated.jsonl", help="Output JSONL path for validated records.")
    parser.add_argument("--out-invalid", "-x", default="src/data_pipeline/dataset/processed/invalid.jsonl", help="Output JSONL path for invalid records.")
    parser.add_argument("--schema-introspection", "-s", default=None, help="Optional default GraphQL introspection JSON file to fallback to.")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of worker threads.")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of records processed (for dev).")
    parser.add_argument("--summary", default="src/data_pipeline/dataset/processed/validate_summary.json", help="Summary JSON output path.")
    args = parser.parse_args()

    summary = process_input(
        input_path=args.input,
        out_validated=args.out_validated,
        out_invalid=args.out_invalid,
        default_schema_introspection=args.schema_introspection,
        workers=args.workers,
        max_records=args.max_records,
    )

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)

    print(f"Validation complete. Processed: {summary['total_processed']}, Valid: {summary['valid']}, Invalid: {summary['invalid']}")
    if summary["invalid"] > 0:
        print("Top error reasons:", summary["error_reasons_top"][:6])
    print(f"Summary written to: {args.summary}")


if __name__ == "__main__":
    main()
