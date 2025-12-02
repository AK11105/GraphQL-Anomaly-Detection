#!/usr/bin/env python3
"""
generate_malicious.py

Generate structured malicious / anomalous GraphQL queries.

Modes:
 - heavy : schema-aware, semantically valid but intentionally expensive (deep, wide, aliases, fragment recursion)
 - fuzz  : schema-agnostic perturbations (invalid fields, random high-entropy names/aliases, malformed but syntactically valid selections)
 - mix   : produce a mix of both

Outputs JSONL records with per-record schema_used in meta (for validator).

Usage example:
 python src/data_pipeline/generators/generate_malicious.py \
   --schemas schema_g1.json,schema_g2.json \
   --out src/data_pipeline/dataset/raw/malicious_raw.jsonl \
   --count 2000 \
   --mode mix \
   --seed 42
"""

from __future__ import annotations
import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from graphql import (
    build_client_schema,
    print_ast,
    GraphQLSchema,
    GraphQLObjectType,
    GraphQLField,
    GraphQLNonNull,
    GraphQLList
)
from graphql.language import OperationType
from graphql.language.ast import (
    DocumentNode,
    OperationDefinitionNode,
    FieldNode,
    NameNode,
    SelectionSetNode,
    ArgumentNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    NamedTypeNode,
    IntValueNode,
    FloatValueNode,
    StringValueNode,
    BooleanValueNode,
    ListValueNode,
    ObjectValueNode,
    ObjectFieldNode,
)

# -----------------------
# Schema helpers (same semantics as generate_normal)
# -----------------------
def load_schema_from_introspection(path: str) -> GraphQLSchema:
    with open(path, "r", encoding="utf-8") as fh:
        intros = json.load(fh)
    return build_client_schema(intros)


def unwrap_type(t):
    while isinstance(t, (GraphQLNonNull, GraphQLList)):
        t = t.of_type
    return t


def is_object_type(t):
    from graphql.type.definition import GraphQLObjectType as _GOT
    return isinstance(t, _GOT)


def is_scalar_type(t):
    from graphql.type.definition import GraphQLScalarType as _GST
    return isinstance(t, _GST)


# -----------------------
# Value node helpers
# -----------------------
def build_value_node(v):
    if isinstance(v, bool):
        return BooleanValueNode(value=v)
    if isinstance(v, int):
        return IntValueNode(value=str(v))
    if isinstance(v, float):
        return FloatValueNode(value=str(v))
    if isinstance(v, str):
        return StringValueNode(value=v)
    if isinstance(v, list):
        return ListValueNode(values=[build_value_node(x) for x in v])
    if isinstance(v, dict):
        return ObjectValueNode(fields=[ObjectFieldNode(name=NameNode(value=k), value=build_value_node(val)) for k, val in v.items()])
    return StringValueNode(value=str(v))


def render_arg_node(name: str, value) -> ArgumentNode:
    return ArgumentNode(name=NameNode(value=name), value=build_value_node(value))


# -----------------------
# Malicious fragment pool
# -----------------------
class MaliciousFragmentPool:
    def __init__(self):
        self.pool: Dict[str, FragmentDefinitionNode] = {}
        self.counter = 0

    def add(self, type_name: str, selection: SelectionSetNode) -> str:
        name = f"MFrag_{type_name}_{self.counter}"
        self.counter += 1
        frag = FragmentDefinitionNode(
            name=NameNode(value=name),
            type_condition=NamedTypeNode(name=NameNode(value=type_name)),
            selection_set=selection,
        )
        self.pool[name] = frag
        return name

    def choose(self, prob: float) -> Optional[str]:
        if not self.pool:
            return None
        if random.random() < prob:
            return random.choice(list(self.pool.keys()))
        return None

    def render_all(self) -> List[FragmentDefinitionNode]:
        return list(self.pool.values())


# -----------------------
# Sampling helpers
# -----------------------
def sample_arg_val_for_type(arg_type):
    base = unwrap_type(arg_type)
    if is_scalar_type(base):
        name = base.name.lower()
        if "int" in name or name == "id":
            return random.randint(1, 10**6)
        if "float" in name:
            return round(random.random() * 1e6, 6)
        if "bool" in name:
            return random.choice([True, False])
        return f"mal_{uuid.uuid4().hex[:8]}"
    return None


def pick_invalid_name():
    # create high-entropy random field name
    return f"x_{uuid.uuid4().hex}"


# -----------------------
# AST field builder
# -----------------------
def make_field_node(name: str, alias: Optional[str] = None, args: Optional[List[ArgumentNode]] = None, selection: Optional[SelectionSetNode] = None) -> FieldNode:
    return FieldNode(
        name=NameNode(value=name),
        alias=NameNode(value=alias) if alias else None,
        arguments=args or [],
        selection_set=selection,
    )


# -----------------------
# Malicious selection generators
# -----------------------
def heavy_selection_for_type(
    gql_type: GraphQLObjectType,
    depth: int,
    max_depth: int,
    fragment_pool: MaliciousFragmentPool,
    alias_bomb: int,
    fragment_recursion_prob: float,
    max_branch: int,
) -> SelectionSetNode:
    """Generate a schema-aware but heavy selection set: deep recursion, wide branching, aliases, fragment recursion."""
    fields = list(gql_type.fields.keys())
    if not fields:
        return SelectionSetNode(selections=[])

    # choose branching wide up to max_branch
    k = min(len(fields), max(1, random.randint(1, max_branch)))
    chosen = random.sample(fields, k=k)

    selections = []
    for fname in chosen:
        f = gql_type.fields[fname]
        # alias-bomb: sometimes add many aliases by sampling additional aliases separately
        alias = None
        if alias_bomb > 0 and random.random() < 0.25:
            # create an alias that looks like an alias bombing entry
            alias = f"a_{uuid.uuid4().hex[:6]}"

        # args: sometimes pass valid args with large or extreme values to increase cost
        args_nodes = []
        for arg_name, arg_def in f.args.items():
            if random.random() < 0.6:
                v = sample_arg_val_for_type(arg_def.type)
                if v is not None:
                    args_nodes.append(render_arg_node(arg_name, v))

        return_type = unwrap_type(f.type)
        if is_object_type(return_type) and depth < max_depth:
            # fragment recursion
            if fragment_pool and random.random() < fragment_recursion_prob:
                inner = heavy_selection_for_type(return_type, depth + 1, max_depth, fragment_pool, alias_bomb, fragment_recursion_prob, max_branch)
                frag_name = fragment_pool.add(return_type.name, inner)
                selections.append(make_field_node(fname, alias=alias, args=args_nodes, selection=SelectionSetNode(selections=[FragmentSpreadNode(name=NameNode(value=frag_name))])))
                continue

            inner = heavy_selection_for_type(return_type, depth + 1, max_depth, fragment_pool, alias_bomb, fragment_recursion_prob, max_branch)
            selections.append(make_field_node(fname, alias=alias, args=args_nodes, selection=inner))
            continue

        selections.append(make_field_node(fname, alias=alias, args=args_nodes, selection=None))

    # optionally add alias bombing extra sibling fields (same field name repeated under different aliases)
    if alias_bomb > 0 and random.random() < 0.15:
        target = random.choice(chosen)
        for _ in range(min(alias_bomb, 6)):
            a = f"ab_{uuid.uuid4().hex[:6]}"
            selections.append(make_field_node(target, alias=a, args=[], selection=None))

    return SelectionSetNode(selections=selections)


def fuzz_selection_for_type(
    gql_type: Optional[GraphQLObjectType],
    depth: int,
    max_depth: int,
    fragment_pool: MaliciousFragmentPool,
    invalid_field_prob: float,
    entropy_injection_prob: float,
    max_branch: int,
) -> SelectionSetNode:
    """
    Schema-agnostic fuzz selection:
    - If gql_type is provided, pick some real fields sometimes
    - Inject invalid fields and high-entropy names
    - Keep syntactically valid AST (no broken braces), but semantic may be invalid
    """
    real_fields = list(gql_type.fields.keys()) if gql_type else []
    k = max(1, random.randint(1, max_branch))
    selections = []

    for _ in range(k):
        use_invalid = random.random() < invalid_field_prob
        if use_invalid or not real_fields:
            name = pick_invalid_name()
            selections.append(make_field_node(name))
            continue

        # pick a real field
        fname = random.choice(real_fields)
        f = gql_type.fields[fname]
        args_nodes = []
        for arg_name, arg_def in f.args.items():
            if random.random() < 0.5:
                v = sample_arg_val_for_type(arg_def.type)
                if v is not None:
                    args_nodes.append(render_arg_node(arg_name, v))

        return_type = unwrap_type(f.type)
        alias = None
        if random.random() < 0.4 and entropy_injection_prob > 0.0:
            alias = f"e_{uuid.uuid4().hex[:5]}"

        if is_object_type(return_type) and depth < max_depth:
            inner = fuzz_selection_for_type(return_type, depth + 1, max_depth, fragment_pool, invalid_field_prob, entropy_injection_prob, max_branch)
            selections.append(make_field_node(fname, alias=alias, args=args_nodes, selection=inner))
            continue

        selections.append(make_field_node(fname, alias=alias, args=args_nodes, selection=None))

    return SelectionSetNode(selections=selections)


# -----------------------
# Document builders (heavy/fuzz/mix)
# -----------------------
def build_malicious_document(
    schema: GraphQLSchema,
    mode: str,
    params: Dict[str, Any],
) -> DocumentNode:
    # choose operation type intentionally: heavy mode may use mutations/subscriptions more
    if mode == "heavy":
        op_choice = random.choices(["query", "mutation", "subscription"], weights=[0.6, 0.3, 0.1], k=1)[0]
    else:
        op_choice = random.choices(["query", "mutation", "subscription"], weights=[0.85, 0.1, 0.05], k=1)[0]

    if op_choice == "query":
        op_type = OperationType.QUERY
        root = schema.query_type
    elif op_choice == "mutation" and schema.mutation_type:
        op_type = OperationType.MUTATION
        root = schema.mutation_type
    elif op_choice == "subscription" and schema.subscription_type:
        op_type = OperationType.SUBSCRIPTION
        root = schema.subscription_type
    else:
        op_type = OperationType.QUERY
        root = schema.query_type

    fragment_pool = MaliciousFragmentPool()
    max_depth = params["max_depth"]
    max_branch = params["max_branch"]

    if mode == "heavy":
        selection = heavy_selection_for_type(root, 1, max_depth, fragment_pool, params["alias_bomb"], params["fragment_recursion_prob"], max_branch)
    elif mode == "fuzz":
        selection = fuzz_selection_for_type(root, 1, max_depth, fragment_pool, params["invalid_field_prob"], params["entropy_injection_prob"], max_branch)
    else:  # mix: randomly choose per-query which behavior to use
        if random.random() < 0.6:
            selection = heavy_selection_for_type(root, 1, max_depth, fragment_pool, params["alias_bomb"], params["fragment_recursion_prob"], max_branch)
        else:
            selection = fuzz_selection_for_type(root, 1, max_depth, fragment_pool, params["invalid_field_prob"], params["entropy_injection_prob"], max_branch)

    # optionally attach a reused fragment spread at top-level to increase recursion
    reused = fragment_pool.choose(params["fragment_reuse_prob"])
    top_selections = list(selection.selections)
    if reused:
        top_selections.append(FragmentSpreadNode(name=NameNode(value=reused)))
        selection = SelectionSetNode(selections=top_selections)

    op = OperationDefinitionNode(
        operation=op_type,
        name=None,
        variable_definitions=[],
        directives=[],
        selection_set=selection,
    )

    definitions = [op] + fragment_pool.render_all()
    return DocumentNode(definitions=definitions)


# -----------------------
# Multi-schema loader and writer
# -----------------------
def load_schemas(schema_inputs: List[str]) -> List[Tuple[GraphQLSchema, str]]:
    schemas = []
    for s in schema_inputs:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"Schema introspection file not found: {s}")
        schema = load_schema_from_introspection(str(p))
        schemas.append((schema, str(p)))
    return schemas


def parse_schema_input(opt: Optional[str]) -> List[str]:
    if not opt:
        return []
    parts = [p.strip() for p in opt.split(",") if p.strip()]
    expanded = []
    for part in parts:
        p = Path(part)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix == ".json":
                    expanded.append(str(f))
        else:
            expanded.append(str(p))
    return expanded


def write_jsonl(path: Path, records: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Generate malicious GraphQL queries (heavy/fuzz/mix)")
    parser.add_argument("--schemas", required=True, help="Comma-separated introspection JSON paths or directory containing .json introspection files")
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=10000, help="Number of queries to generate")
    parser.add_argument("--mode", choices=["heavy", "fuzz", "mix"], default="mix", help="Type of malicious generation")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-depth", type=int, default=12, help="Max depth for heavy mode (use high values for deep attacks)")
    parser.add_argument("--max-branch", type=int, default=6, help="Max branching factor")
    parser.add_argument("--alias-bomb", type=int, default=4, help="Number of alias-bomb siblings when triggered")
    parser.add_argument("--fragment-recursion-prob", type=float, default=0.35, help="Probability to create recursive fragments")
    parser.add_argument("--fragment-reuse-prob", type=float, default=0.3, help="Probability to reuse existing fragment spread")
    parser.add_argument("--invalid-field-prob", type=float, default=0.35, help="For fuzz mode: probability to inject invalid field names")
    parser.add_argument("--entropy-injection-prob", type=float, default=0.4, help="For fuzz mode: probability to inject high-entropy aliases/names")
    parser.add_argument("--schema-weights", default=None, help="Optional comma-separated weights matching schemas order")
    args = parser.parse_args()

    random.seed(args.seed)

    schema_paths = parse_schema_input(args.schemas)
    if not schema_paths:
        raise SystemExit("No schema files found. Provide --schemas path(s)")

    schemas = load_schemas(schema_paths)
    schema_weights = None
    if args.schema_weights:
        schema_weights = [float(x) for x in args.schema_weights.split(",")]
        if len(schema_weights) != len(schemas):
            raise SystemExit("schema_weights length must match number of schemas")

    params = {
        "max_depth": args.max_depth,
        "max_branch": args.max_branch,
        "alias_bomb": args.alias_bomb,
        "fragment_recursion_prob": args.fragment_recursion_prob,
        "fragment_reuse_prob": args.fragment_reuse_prob,
        "invalid_field_prob": args.invalid_field_prob,
        "entropy_injection_prob": args.entropy_injection_prob,
    }

    records = []
    for i in range(args.count):
        idx = random.choices(range(len(schemas)), weights=schema_weights, k=1)[0] if schema_weights else random.randrange(len(schemas))
        schema, path = schemas[idx]
        doc = build_malicious_document(schema, args.mode, params)
        q_text = print_ast(doc)

        rec = {
            "query_id": str(uuid.uuid4()),
            "query": q_text,
            "type": "malicious",
            "meta": {
                "generator": "malicious_ast",
                "generator_params": {
                    "mode": args.mode,
                    "max_depth": args.max_depth,
                    "max_branch": args.max_branch,
                    "alias_bomb": args.alias_bomb,
                    "fragment_recursion_prob": args.fragment_recursion_prob,
                    "fragment_reuse_prob": args.fragment_reuse_prob,
                    "invalid_field_prob": args.invalid_field_prob,
                    "entropy_injection_prob": args.entropy_injection_prob,
                    "schema_used": path,
                    "seed": args.seed,
                },
            },
        }
        records.append(rec)

    write_jsonl(Path(args.out), records)
    print(f"Generated {len(records)} malicious queries -> {args.out}")


if __name__ == "__main__":
    main()
