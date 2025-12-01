// src/ingestion/parser/cost_calc.js

import {
  Kind,
  GraphQLObjectType,
  getNamedType
} from "graphql";

/**
 * Phase-1 structural cost engine.
 * Works without external libs.
 * Expands fragments.
 * Validates fields using schema.
 * Computes meaningful cost used by tests.
 */

export async function computeCostMetrics({ schema, ast }) {
  if (!schema) {
    return {
      estimated_cost: null,
      complexity_score: null,
      error: "Schema not provided"
    };
  }

  if (!ast) {
    return {
      estimated_cost: null,
      complexity_score: null,
      error: "AST is null"
    };
  }

  const fragmentMap = Object.create(null);

  // ---------------------------------------------------
  // Build FragmentDefinition map
  // ---------------------------------------------------
  for (const def of ast.definitions) {
    if (def.kind === Kind.FRAGMENT_DEFINITION) {
      fragmentMap[def.name.value] = def;
    }
  }

  const rootType = schema.getQueryType();
  let totalFields = 0;
  let maxDepth = 1;
  let error = null;

  // ---------------------------------------------------
  // Field validation helper
  // ---------------------------------------------------
  function getFieldDef(parentType, fieldName) {
    if (!(parentType instanceof GraphQLObjectType)) return null;
    return parentType.getFields()[fieldName] || null;
  }

  // ---------------------------------------------------
  // Walk selection sets recursively
  // ---------------------------------------------------
  function walkSelection(node, parentType, depth) {
    if (error) return;

    // ---- FIELD --------------------------------------
    if (node.kind === Kind.FIELD) {
      const name = node.name.value;

      const fieldDef = getFieldDef(parentType, name);
      if (!fieldDef) {
        error = `Unknown field '${name}' on type '${parentType?.name}'`;
        return;
      }

      totalFields += 1;
      const nextType = getNamedType(fieldDef.type);

      if (node.selectionSet) {
        maxDepth = Math.max(maxDepth, depth + 1);

        for (const child of node.selectionSet.selections) {
          walkSelection(child, nextType, depth + 1);
        }
      }

      return;
    }

    // ---- FRAGMENT SPREAD -----------------------------
    if (node.kind === Kind.FRAGMENT_SPREAD) {
      const fragName = node.name.value;
      const frag = fragmentMap[fragName];
      if (!frag) {
        error = `Unknown fragment '${fragName}'`;
        return;
      }
      walkSelection(frag.selectionSet, parentType, depth);
      return;
    }

    // ---- FRAGMENT DEFINITION -------------------------
    if (node.kind === Kind.FRAGMENT_DEFINITION) {
      walkSelection(node.selectionSet, parentType, depth);
      return;
    }

    // ---- SELECTION SET -------------------------------
    if (node.kind === Kind.SELECTION_SET) {
      for (const sel of node.selections) {
        walkSelection(sel, parentType, depth);
      }
    }
  }

  // ---------------------------------------------------
  // Walk each operation root
  // ---------------------------------------------------
  for (const def of ast.definitions) {
    if (def.kind === Kind.OPERATION_DEFINITION) {
      walkSelection(def.selectionSet, rootType, 1);
    }
  }

  // Invalid field → return test-required state
  if (error) {
    return {
      estimated_cost: null,
      complexity_score: null,
      error
    };
  }

  // ---------------------------------------------------
  // Final meaningful Phase-1 cost
  // ---------------------------------------------------
  const estimatedCost = totalFields + maxDepth * 1.5;

  return {
    estimated_cost: estimatedCost,
    complexity_score: estimatedCost,
    error: null
  };
}
