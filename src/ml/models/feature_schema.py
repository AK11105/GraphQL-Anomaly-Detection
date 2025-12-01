# ml/models/feature_schema.py
"""
Feature schema used across preprocessing, training, and serving.
Keep this in sync with your ingestion / feature_extractor implementation.
"""

from typing import List

STATIC_NUMERIC_FEATURES: List[str] = [
    "depth",
    "cost",
    "token_count",
    "unique_field_count",
    "avg_field_repetition",
    "max_field_repetition",
    "entropy_chars",
    "entropy_tokens",
    "fragment_count",
    "alias_count",
    "directive_count",
]

STATIC_BINARY_FEATURES: List[str] = [
    "has_recursive_fragments",
    "has_deep_alias_chains",
    "has_introspection_usage",
    "has_field_repetition",
    "has_overlapping_fragments",
]

# Full feature vector order used by models: numeric first, then binary as floats (0.0/1.0)
FEATURE_ORDER: List[str] = STATIC_NUMERIC_FEATURES + STATIC_BINARY_FEATURES

FEATURE_DIM: int = len(FEATURE_ORDER)
