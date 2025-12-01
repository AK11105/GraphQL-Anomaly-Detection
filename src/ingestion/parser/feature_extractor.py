import json
import math
import subprocess
from pathlib import Path


NODE_RUNNER = str(Path(__file__).parent / "run_metrics.js")


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    length = len(text)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def extract_features(query: str, schema_sdl: str = None) -> dict:
    """
    Master Phase-1 feature extraction.
    Combines: AST + depth + cost + entropy + length + counters.
    """

    if not isinstance(query, str) or not query.strip():
        return {"error": "Invalid query string", "features": None}

    # ---------------------------
    # 1. Call Node aggregator
    # ---------------------------
    payload = {
        "query": query,
        "schemaSDL": schema_sdl,
    }

    try:
        proc = subprocess.Popen(
            ["node", NODE_RUNNER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(json.dumps(payload))
    except Exception as e:
        return {"error": f"Node execution failed: {e}", "features": None}

    if stderr:
        # Node may log warnings; not always fatal
        print("NODE STDERR:", stderr)
        pass

    try:
        result = json.loads(stdout)
    except Exception:
        return {"error": "Invalid JSON returned from Node", "features": None}

    ast_info = result.get("ast") or {}
    depth_info = result.get("depth") or {}
    cost_info = result.get("cost") or {}

    # ---------------------------
    # 2. Python-only metrics
    # ---------------------------
    entropy_value = shannon_entropy(query)
    length_chars = len(query)
    num_tokens = len(query.replace("\n", " ").split())

    # ---------------------------
    # 3. Merge final feature vector
    # ---------------------------
    features = {
        # Structural
        "num_fields": ast_info["stats"].get("num_fields"),
        "num_fragments": ast_info["stats"].get("num_fragments"),
        "num_directives": ast_info["stats"].get("num_directives"),
        "num_aliases": ast_info["stats"].get("num_aliases"),
        "num_operations": ast_info["stats"].get("num_operations"),
        "num_mutations": ast_info["stats"].get("num_mutations"),
        "num_subscriptions": ast_info["stats"].get("num_subscriptions"),
        "num_variables": ast_info["stats"].get("num_variables"),
        "num_arguments": ast_info["stats"].get("num_arguments"),
        "num_introspection_ops": ast_info["stats"].get("num_introspection_operations"),

        # Depth
        "query_depth": depth_info.get("query_depth"),
        "avg_depth": depth_info.get("avg_depth"),
        "branching_factor": depth_info.get("branching_factor"),
        "node_count": depth_info.get("node_count"),
        "num_nested_selections": depth_info.get("num_nested_selections"),

        # Cost
        "estimated_cost": cost_info.get("estimated_cost"),
        "complexity_score": cost_info.get("complexity_score"),

        # Text-level
        "entropy": entropy_value,
        "query_length": length_chars,
        "num_tokens": num_tokens,

        # Flags
        "has_error": ast_info.get("error") is not None
                    or cost_info.get("error") is not None
    }

    return {
        "error": None,
        "features": features,
        "ast_error": ast_info.get("error"),
        "cost_error": cost_info.get("error"),
    }
