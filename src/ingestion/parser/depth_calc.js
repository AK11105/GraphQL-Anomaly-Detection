// src/ingestion/parser/depth_calc.js

import { visit } from "graphql";

/**
 * Compute depth-related metrics for a parsed GraphQL AST.
 * Phase-1: Pure structural metrics (no schema required).
 *
 * Outputs:
 * - query_depth (max depth)
 * - avg_depth
 * - branching_factor (MAX branching among all nodes)
 * - node_count
 * - num_nested_selections
 */

export function computeDepthMetrics(ast) {
  if (!ast) {
    return {
      query_depth: 0,
      avg_depth: 0,
      branching_factor: 0,
      node_count: 0,
      num_nested_selections: 0,
    };
  }

  let maxDepth = 0;
  let depthSum = 0;
  let nodeCount = 0;

  let maxBranching = 0;
  let numNestedSelections = 0;

  /**
   * DFS Walker
   */
  function walk(node, depth = 1) {
    nodeCount += 1;
    depthSum += depth;

    if (depth > maxDepth) {
      maxDepth = depth;
    }

    if (depth > 1) {
      numNestedSelections += 1;
    }

    const selections = node.selectionSet?.selections || [];

    // MAX branching factor for anomaly detection
    if (selections.length > maxBranching) {
      maxBranching = selections.length;
    }

    for (const child of selections) {
      walk(child, depth + 1);
    }
  }

  // Start DFS at each operation root
  visit(ast, {
    OperationDefinition(node) {
      walk(node, 1);
    },
  });

  const avgDepth = nodeCount > 0 ? depthSum / nodeCount : 0;

  return {
    query_depth: maxDepth,
    avg_depth: avgDepth,
    branching_factor: maxBranching,   // UPDATED
    node_count: nodeCount,
    num_nested_selections: numNestedSelections,
  };
}
