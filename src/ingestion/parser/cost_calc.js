// src/ingestion/parser/cost_calc.js

import {
  Kind,
  GraphQLObjectType,
  getNamedType
} from "graphql";

/**
 * Phase-1 structural cost engine.
 * - Validates fields using schema
 * - Expands fragments using correct typeCondition
 * - Resolves root types correctly for query/mutation/subscription
 * - Computes totalFields + depth * 1.5
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
  // Collect FragmentDefinitions
  // ---------------------------------------------------
  for (const def of ast.definitions) {
    if (def.kind === Kind.FRAGMENT_DEFINITION) {
      fragmentMap[def.name.value] = def;
    }
  }

  let totalFields = 0;
  let maxDepth = 1;
  let error = null;

  // ---------------------------------------------------
  // Resolve schema root type per operation
  // ---------------------------------------------------
  function getOperationRoot(def) {
    if (!def.operation) return schema.getQueryType();

    switch (def.operation) {
      case "query":
        return schema.getQueryType();
      case "mutation":
        return schema.getMutationType() || schema.getQueryType();
      case "subscription":
        return schema.getSubscriptionType() || schema.getQueryType();
      default:
        return schema.getQueryType();
    }
  }

  // ---------------------------------------------------
  // Field validation
  // ---------------------------------------------------
  function getFieldDef(parentType, fieldName) {
    if (!(parentType instanceof GraphQLObjectType)) return null;
    return parentType.getFields()[fieldName] || null;
  }

  // ---------------------------------------------------
  // Recursive walker
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

      const typeName = frag.typeCondition.name.value;
      const fragType = schema.getType(typeName);
      if (!fragType) {
        error = `Unknown type '${typeName}' for fragment '${fragName}'`;
        return;
      }

      walkSelection(frag.selectionSet, fragType, depth);
      return;
    }

    // ---- INLINE / FRAGMENT DEFINITION ----------------
    if (node.kind === Kind.FRAGMENT_DEFINITION) {
      const typeName = node.typeCondition.name.value;
      const fragType = schema.getType(typeName);
      if (!fragType) {
        error = `Unknown type '${typeName}' in fragment definition`;
        return;
      }

      walkSelection(node.selectionSet, fragType, depth);
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
  // Traverse each operation
  // ---------------------------------------------------
  for (const def of ast.definitions) {
    if (def.kind === Kind.OPERATION_DEFINITION) {
      const opRoot = getOperationRoot(def);

      if (!opRoot) {
        error = `Schema missing root type for operation '${def.operation}'`;
        break;
      }

      walkSelection(def.selectionSet, opRoot, 1);
    }
  }

  if (error) {
    return {
      estimated_cost: null,
      complexity_score: null,
      error
    };
  }

  // ---------------------------------------------------
  // Final cost
  // ---------------------------------------------------
  const estimatedCost = totalFields + maxDepth * 1.5;

  return {
    estimated_cost: estimatedCost,
    complexity_score: estimatedCost,
    error: null
  };
}
