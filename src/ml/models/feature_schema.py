# src/ml/models/feature_schema.py

"""
Central feature schema (Phase-1 complete).

Includes:
 - Structural AST statistics (from ast_extractor.js)
 - Depth metrics (from depth_calc.js)
 - Cost metrics (from cost_calc.js)
 - Text-level metrics (entropy, length, tokens)
 - Error flags

This list should only grow in later phases—never reorder or remove keys.
"""

FEATURE_KEYS = [
    # -----------------------------
    # AST / Structural Metrics
    # -----------------------------
    "num_fields",
    "num_fragments",
    "num_directives",
    "num_aliases",
    "num_operations",
    "num_mutations",
    "num_subscriptions",
    "num_variables",
    "num_arguments",
    "num_introspection_ops",

    # -----------------------------
    # Depth Metrics
    # -----------------------------
    "query_depth",
    "avg_depth",
    "branching_factor",
    "node_count",
    "num_nested_selections",

    # -----------------------------
    # Cost Metrics
    # -----------------------------
    "estimated_cost",
    "complexity_score",

    # -----------------------------
    # Text-Level Metrics
    # -----------------------------
    "entropy",
    "query_length",
    "num_tokens",

    # -----------------------------
    # Error Flags
    # -----------------------------
    "has_error",
]


def get_feature_keys():
    """Return stable ordered list of Phase-1 feature keys."""
    return FEATURE_KEYS.copy()
