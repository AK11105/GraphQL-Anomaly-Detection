# tests/feature_extractor.test.py
import sys
print(sys.path)


import json
import pytest
from src.ingestion.parser.feature_extractor import extract_features


# ------------------------------------------------------
# Minimal test schema for cost calculation
# ------------------------------------------------------

TEST_SCHEMA = """
type Query {
    user(id: ID!): User
    posts(limit: Int, offset: Int): [Post]
    simple: String
}

type User {
    id: ID
    name: String
    details: UserDetails
}

type UserDetails {
    email: String
    address: String
}

type Post {
    title: String
    body: String
}
"""


# ------------------------------------------------------
# Helper: Ensure all required keys exist
# ------------------------------------------------------

REQUIRED_FEATURE_KEYS = {
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
    "query_depth",
    "avg_depth",
    "branching_factor",
    "node_count",
    "num_nested_selections",
    "estimated_cost",
    "complexity_score",
    "entropy",
    "query_length",
    "num_tokens",
    "has_error",
}


def assert_feature_keys(features: dict):
    missing = REQUIRED_FEATURE_KEYS - features.keys()
    assert not missing, f"Missing keys: {missing}"


# ------------------------------------------------------
# Tests
# ------------------------------------------------------

def test_basic_query_features():
    query = """
        query {
            simple
        }
    """
    result = extract_features(query, TEST_SCHEMA)

    assert result["error"] is None
    features = result["features"]
    assert_feature_keys(features)

    assert features["num_fields"] == 1
    assert features["query_length"] > 0
    assert features["entropy"] > 0
    assert features["estimated_cost"] >= 1


def test_nested_query_features():
    query = """
        query {
            user(id: 1) {
                details {
                    email
                    address
                }
            }
        }
    """
    result = extract_features(query, TEST_SCHEMA)
    f = result["features"]

    assert f["num_fields"] >= 3
    assert f["query_depth"] >= 3
    assert f["estimated_cost"] >= 3


def test_alias_and_argument_features():
    query = """
        query {
            posts(limit: 10, offset: 2) {
                title
            }
        }
    """
    result = extract_features(query, TEST_SCHEMA)
    f = result["features"]

    assert f["num_aliases"] == 0
    assert f["num_arguments"] == 2
    assert f["query_depth"] >= 3


def test_fragment_query_features():
    query = """
        query {
            user(id: 1) {
                ...UserFields
            }
        }

        fragment UserFields on User {
            id
            name
        }
    """

    result = extract_features(query, TEST_SCHEMA)
    f = result["features"]

    assert f["num_fragments"] >= 1
    assert f["num_fields"] >= 1    # user + fragment spread
    assert f["query_depth"] >= 2


def test_introspection_query():
    query = """
        {
            __schema {
                types {
                    name
                }
            }
        }
    """

    result = extract_features(query, TEST_SCHEMA)
    f = result["features"]

    assert f["num_introspection_ops"] >= 1
    assert f["num_fields"] >= 2
    assert f["query_depth"] >= 3


def test_invalid_query_returns_error_flag():
    query = "query { invalidField }"

    result = extract_features(query, TEST_SCHEMA)
    f = result["features"]

    # AST extractor fails → has_error must be True
    assert f["has_error"] is True


def test_schema_missing_disables_cost():
    query = """
        query {
            simple
        }
    """
    result = extract_features(query, schema_sdl=None)
    f = result["features"]

    assert f["estimated_cost"] is None
    assert f["complexity_score"] is None
    assert f["has_error"] in (False, True)  # allowed both


def test_multiple_operations():
    query = """
        query A { simple }
        query B { user(id: 1) { id } }
    """

    result = extract_features(query, TEST_SCHEMA)
    f = result["features"]

    assert f["num_operations"] == 2
    assert f["query_depth"] >= 2
    assert f["estimated_cost"] >= 1


def test_entropy_and_length_sanity():
    query = "{ simple }"

    result = extract_features(query, TEST_SCHEMA)
    f = result["features"]

    assert f["query_length"] == len(query)
    assert f["entropy"] > 0
    assert f["num_tokens"] >= 1
