#!/usr/bin/env python3
"""
generate_normal.py (AST-backed final, per-record-schema)

Generates schema-valid GraphQL queries by building graphql-core AST nodes and printing them with print_ast.
Outputs JSONL records:
{
  "query_id": "...",
  "query": "...",
  "type": "normal",
  "meta": {
    "generator": "normal_ast",
    "generator_params": { "schema_used": "schema_g2.json", ... }
  }
}

This generator produces strictly schema-valid queries (uses the schema's field names).
It supports:
- Multi-schema input (comma list or directory)
- Generalization mode (diverse shapes)
- Uniform or weighted field sampling
- Occasional fragments (per-query fragment pool)
- Deterministic with --seed
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
    GraphQLList,
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
# Schema helpers
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
# Value node builders
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
# Sampling helpers
# -----------------------
def choose_num_children(max_avail: int, generalize: bool) -> int:
    if max_avail <= 0:
        return 0
    if generalize:
        return max(1, min(max_avail, int(random.expovariate(1.0)) + 1))
    else:
        return max(1, min(max_avail, int(random.expovariate(1.3)) + 1))


def sample_arg_value(arg_type):
    base = unwrap_type(arg_type)
    if is_scalar_type(base):
        name = base.name.lower()
        if "int" in name or name == "id":
            return random.randint(1, 9999)
        if "float" in name:
            return round(random.random() * 1000, 3)
        if "bool" in name:
            return random.choice([True, False])
        return f"s_{uuid.uuid4().hex[:6]}"
    return None


# -----------------------
# Fragment pool (AST fragments)
# -----------------------
class ASTFragmentPool:
    def __init__(self):
        self.pool: Dict[str, FragmentDefinitionNode] = {}
        self.counter = 0

    def add(self, type_name: str, selection: SelectionSetNode) -> str:
        name = f"F_{type_name}_{self.counter}"
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
# Selection builders (AST)
# -----------------------
def make_field_node(name: str, alias: Optional[str] = None, args: Optional[List[ArgumentNode]] = None, selection: Optional[SelectionSetNode] = None) -> FieldNode:
    return FieldNode(
        name=NameNode(value=name),
        alias=NameNode(value=alias) if alias else None,
        arguments=args or [],
        selection_set=selection,
    )


def build_selection_set_for_type(
    gql_type: GraphQLObjectType,
    depth: int,
    max_depth: int,
    fragment_pool: ASTFragmentPool,
    fragment_prob: float,
    alias_prob: float,
    arg_prob: float,
    generalize: bool,
    uniform_field_sampling: bool,
) -> SelectionSetNode:
    fields: Dict[str, GraphQLField] = gql_type.fields
    names = list(fields.keys())
    if not names:
        return SelectionSetNode(selections=[])

    if uniform_field_sampling:
        k = choose_num_children(len(names), generalize)
        chosen = random.sample(names, k=k)
    else:
        weights = []
        for fname in names:
            f = fields[fname]
            rt = unwrap_type(f.type)
            w = 1.0
            if is_object_type(rt):
                w += 0.6
            if getattr(f, "args", None):
                w += 0.2
            weights.append(w)
        total = sum(weights)
        probs = [w / total for w in weights]
        k = choose_num_children(len(names), generalize)
        chosen = []
        available = names.copy()
        avail_probs = probs.copy()
        for _ in range(min(k, len(available))):
            idx = random.choices(range(len(available)), weights=avail_probs, k=1)[0]
            chosen.append(available.pop(idx))
            avail_probs.pop(idx)

    selections = []
    for fname in chosen:
        f = fields[fname]
        args_nodes = []
        for arg_name, arg_def in f.args.items():
            if random.random() < arg_prob:
                val = sample_arg_value(arg_def.type)
                if val is not None:
                    args_nodes.append(render_arg_node(arg_name, val))
        alias = f"a_{uuid.uuid4().hex[:6]}" if random.random() < alias_prob else None

        return_type = unwrap_type(f.type)
        if is_object_type(return_type) and depth < max_depth:
            # fragment option
            if fragment_pool and random.random() < fragment_prob:
                inner_sel = build_selection_set_for_type(return_type, depth + 1, max_depth, fragment_pool, fragment_prob, alias_prob, arg_prob, generalize, uniform_field_sampling)
                frag_name = fragment_pool.add(return_type.name, inner_sel)
                spread = FragmentSpreadNode(name=NameNode(value=frag_name))
                sel = SelectionSetNode(selections=[spread])
                node = make_field_node(fname, alias=alias, args=args_nodes, selection=sel)
                selections.append(node)
                continue
            inner_sel = build_selection_set_for_type(return_type, depth + 1, max_depth, fragment_pool, fragment_prob, alias_prob, arg_prob, generalize, uniform_field_sampling)
            node = make_field_node(fname, alias=alias, args=args_nodes, selection=inner_sel)
            selections.append(node)
            continue

        node = make_field_node(fname, alias=alias, args=args_nodes, selection=None)
        selections.append(node)

    return SelectionSetNode(selections=selections)


# -----------------------
# Operation & document builders
# -----------------------
def build_operation_document(
    schema: GraphQLSchema,
    min_depth: int,
    max_depth: int,
    fragment_pool: ASTFragmentPool,
    params: Dict[str, Any],
    generalize: bool,
    uniform_field_sampling: bool,
) -> DocumentNode:
    op_choice = random.choices(["query", "mutation", "subscription"], weights=[0.9, 0.08, 0.02], k=1)[0]
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

    depth = random.randint(min_depth, max_depth)
    selection_set = build_selection_set_for_type(root, depth=1, max_depth=depth, fragment_pool=fragment_pool, fragment_prob=params["fragment_prob"], alias_prob=params["alias_prob"], arg_prob=params["arg_prob"], generalize=generalize, uniform_field_sampling=uniform_field_sampling)

    reused = fragment_pool.choose(params["fragment_reuse_prob"])
    top_selections = list(selection_set.selections)
    if reused:
        top_selections.append(FragmentSpreadNode(name=NameNode(value=reused)))
        selection_set = SelectionSetNode(selections=top_selections)

    operation = OperationDefinitionNode(
        operation=op_type,
        name=None,
        variable_definitions=[],
        directives=[],
        selection_set=selection_set,
    )

    fragments = fragment_pool.render_all()
    definitions = [operation] + fragments
    return DocumentNode(definitions=definitions)


# -----------------------
# Multi-schema loader
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


def choose_schema_index(schemas_len: int, weights: Optional[List[float]]) -> int:
    if weights:
        return random.choices(range(schemas_len), weights=weights, k=1)[0]
    return random.randrange(schemas_len)


# -----------------------
# JSONL writer & CLI
# -----------------------
def write_jsonl(path: Path, records: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="AST-backed normal GraphQL query generator (per-record-schema)")
    parser.add_argument("--schemas", required=True, help="Comma-separated introspection JSON paths or directory containing .json introspection files")
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=10000, help="Number of queries to generate")
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--generalize", action="store_true", help="Enable generalization mode")
    parser.add_argument("--uniform-field-sampling", action="store_true", help="Sample fields uniformly within a type")
    parser.add_argument("--fragment-prob", type=float, default=0.12, help="Probability to create a fragment for an object field (occasional)")
    parser.add_argument("--fragment-reuse-prob", type=float, default=0.15, help="Probability to reuse an existing fragment")
    parser.add_argument("--alias-prob", type=float, default=0.08, help="Probability to use alias on a field")
    parser.add_argument("--arg-prob", type=float, default=0.25, help="Probability to populate an argument on a field")
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

    records = []
    for i in range(args.count):
        idx = choose_schema_index(len(schemas), schema_weights)
        schema, path = schemas[idx]
        fragment_pool = ASTFragmentPool()  # per-query fragments
        params = {
            "fragment_prob": args.fragment_prob,
            "fragment_reuse_prob": args.fragment_reuse_prob,
            "alias_prob": args.alias_prob,
            "arg_prob": args.arg_prob,
        }
        doc = build_operation_document(
            schema=schema,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            fragment_pool=fragment_pool,
            params=params,
            generalize=args.generalize,
            uniform_field_sampling=args.uniform_field_sampling,
        )
        q_text = print_ast(doc)
        rec = {
            "query_id": str(uuid.uuid4()),
            "query": q_text,
            "type": "normal",
            "meta": {
                "generator": "normal_ast",
                "generator_params": {
                    "min_depth": args.min_depth,
                    "max_depth": args.max_depth,
                    "generalize": args.generalize,
                    "uniform_field_sampling": args.uniform_field_sampling,
                    "schema_used": path,
                    "seed": args.seed,
                },
            },
        }
        records.append(rec)

    write_jsonl(Path(args.out), records)
    print(f"Generated {len(records)} queries -> {args.out}")


if __name__ == "__main__":
    main()
