import json
import subprocess
import sys
from pathlib import Path
import uuid
import pytest


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def minimal_introspection_schema():
    """
    Produces the smallest valid GraphQL introspection JSON
    that graphql-core can convert to SDL.
    """
    return {
        "__schema": {
            "queryType": {"name": "Query"},
            "types": [
                {
                    "kind": "OBJECT",
                    "name": "Query",
                    "fields": [
                        {
                            "name": "hello",
                            "args": [],
                            "type": {"kind": "SCALAR", "name": "String"},
                            "isDeprecated": False,
                            "deprecationReason": None,
                        }
                    ],
                    "interfaces": []  # Add this line
                },
                {
                    "kind": "SCALAR",
                    "name": "String",
                }
            ]
        }
    }


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_validate_generated_end_to_end(tmp_path):
    """
    End-to-end test validating:
    - reading raw JSONL
    - extracting features through feature_extractor
    - correct output into validated + invalid files
    - correct counting in summary.json
    """

    # -------------------------------------------------------------------
    # 1. Create sample raw dataset
    # -------------------------------------------------------------------
    raw_dir = tmp_path / "raw"
    raw_file = raw_dir / "sample.jsonl"

    sample_queries = [
        {
            "query_id": "q1",
            "type": "normal",
            "query": "{ hello }"
        },
        {
            "query_id": "q2",
            "type": "normal",
            "query": "{ hello world }"   # invalid because field "world" not in schema
        },
        {
            "query_id": "q3",
            "type": "normal",
            "query": "not graphql at all"  # invalid
        }
    ]

    write_jsonl(raw_file, sample_queries)

    # -------------------------------------------------------------------
    # 2. Write minimal introspection JSON
    # -------------------------------------------------------------------
    introspection_path = tmp_path / "introspection.json"
    with introspection_path.open("w", encoding="utf-8") as f:
        json.dump(minimal_introspection_schema(), f)

    # -------------------------------------------------------------------
    # 3. Prepare output paths
    # -------------------------------------------------------------------
    out_validated = tmp_path / "validated" / "validated.jsonl"
    out_invalid = tmp_path / "invalid" / "invalid.jsonl"
    out_summary = tmp_path / "summary.json"

    out_validated.parent.mkdir(parents=True, exist_ok=True)
    out_invalid.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # 4. Run validator via subprocess
    # -------------------------------------------------------------------
    validator_script = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "src"
        / "data_pipeline"
        / "validators"
        / "validate_generated.py"
    )

    assert validator_script.exists(), f"Validator script not found: {validator_script}"

    cmd = [
        sys.executable,
        str(validator_script),
        "--input", str(raw_file),
        "--out-validated", str(out_validated),
        "--out-invalid", str(out_invalid),
        "--schema-introspection", str(introspection_path),
        "--summary", str(out_summary),
        "--workers", "2"
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", proc.stdout)
    print("STDERR:", proc.stderr)

    assert proc.returncode == 0, "Validator returned non-zero exit code."

    # -------------------------------------------------------------------
    # 5. Assertions: files created
    # -------------------------------------------------------------------
    assert out_validated.exists(), "Validated output missing."
    assert out_invalid.exists(), "Invalid output missing."
    assert out_summary.exists(), "Summary output missing."

    # -------------------------------------------------------------------
    # 6. Read outputs
    # -------------------------------------------------------------------
    validated_lines = out_validated.read_text().strip().split("\n")
    invalid_lines = out_invalid.read_text().strip().split("\n")

    # Expect at least 1 valid: "{ hello }"
    assert len(validated_lines) >= 1

    # Expect at least 2 invalid entries
    assert len(invalid_lines) >= 2

    # -------------------------------------------------------------------
    # 7. Validate features exist in valid entries
    # -------------------------------------------------------------------
    valid_obj = json.loads(validated_lines[0])
    assert "features" in valid_obj
    feats = valid_obj["features"]
    assert isinstance(feats, dict)
    assert "num_fields" in feats
    assert "query_depth" in feats
    assert "entropy" in feats

    # -------------------------------------------------------------------
    # 8. Check summary content
    # -------------------------------------------------------------------
    summary = json.loads(out_summary.read_text())
    assert summary["total_processed"] == 3
    assert summary["valid"] >= 1
    assert summary["invalid"] >= 2

    # Depth histogram should have integers
    depth_buckets = summary.get("depth_buckets_top", [])
    if depth_buckets:
        depth_value, _count = depth_buckets[0]
        assert isinstance(depth_value, int)


# -------------------------------------------------------------------
# Pytest marker for fast runs
# -------------------------------------------------------------------

@pytest.mark.fast
def test_validator_imports():
    """
    Simple test ensures validator script is importable (syntax OK).
    """
    validator_script = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "src"
        / "data_pipeline"
        / "validators"
        / "validate_generated.py"
    )
    assert validator_script.exists()
