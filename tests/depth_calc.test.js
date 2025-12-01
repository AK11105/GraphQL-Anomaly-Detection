// tests/depth_calc.test.js

import { describe, test, expect } from "vitest";
import { parse } from "graphql";
import { computeDepthMetrics } from "../src/ingestion/parser/depth_calc.js";

// Utility to parse safely
function ast(query) {
  return parse(query);
}

describe("Depth Calculator - Phase 1 (Complete Coverage)", () => {
  // ------------------------------------------------------
  test("computes correct depth/branching for a simple query", () => {
    const query = `
      query {
        user {
          id
          name
        }
      }
    `;

    const result = computeDepthMetrics(ast(query));

    expect(result.query_depth).toBe(3); // query -> user -> fields
    expect(result.node_count).toBeGreaterThan(3);
    expect(result.num_nested_selections).toBeGreaterThan(1);
    expect(result.branching_factor).toBeGreaterThan(0);
    expect(result.avg_depth).toBeGreaterThan(1);
  });

  // ------------------------------------------------------
  test("deeply nested selections compute correct max depth", () => {
    const query = `
      query {
        a {
          b {
            c {
              d {
                e {
                  id
                }
              }
            }
          }
        }
      }
    `;

    const result = computeDepthMetrics(ast(query));

    // Depth levels:
    // 1: query
    // 2: a
    // 3: b
    // 4: c
    // 5: d
    // 6: e
    // 7: id
    expect(result.query_depth).toBe(7);
    expect(result.num_nested_selections).toBeGreaterThan(5);
  });

  // ------------------------------------------------------
  test("branching factor reflects number of children correctly", () => {
    const query = `
      query {
        root {
          x
          y
          z
        }
      }
    `;

    const result = computeDepthMetrics(ast(query));

    // root has 3 children → branching factor ~3 for that parent node
    expect(result.branching_factor).toBe(3);
  });

  // ------------------------------------------------------
  test("fragments do not break depth calculation", () => {
    const query = `
      query {
        user {
          ...UserFields
        }
      }

      fragment UserFields on User {
        id
        profile {
          name
        }
      }
    `;

    const result = computeDepthMetrics(ast(query));

    // Depth from selectionSet only; spreads are counted as nodes but not structural children
    expect(result.query_depth).toBeGreaterThanOrEqual(3);
    expect(result.node_count).toBe(3);
  });

  // ------------------------------------------------------
  test("aliases do not affect depth but increase node count", () => {
    const query = `
      query {
        user {
          profile: details {
            a
          }
        }
      }
    `;

    const result = computeDepthMetrics(ast(query));

    expect(result.query_depth).toBe(4); // query → user → details → a
    expect(result.node_count).toBeGreaterThan(3);
  });

  // ------------------------------------------------------
  test("arguments do not change depth", () => {
    const query = `
      query {
        posts(limit: 10, offset: 20) {
          title
        }
      }
    `;

    const result = computeDepthMetrics(ast(query));

    // Same depth as if no args:
    expect(result.query_depth).toBe(3);
  });

  // ------------------------------------------------------
  test("multiple operations compute depth for each root", () => {
    const query = `
      query Q1 {
        a { b }
      }

      query Q2 {
        x {
          y {
            z
          }
        }
      }
    `;

    const result = computeDepthMetrics(ast(query));

    // Q1 max depth = 3
    // Q2 max depth = 4
    // Result should reflect the MAX depth across operations:
    expect(result.query_depth).toBe(4);
  });

  // ------------------------------------------------------
  test("empty AST or null returns zeroed metrics", () => {
    const result = computeDepthMetrics(null);

    expect(result.query_depth).toBe(0);
    expect(result.avg_depth).toBe(0);
    expect(result.branching_factor).toBe(0);
    expect(result.node_count).toBe(0);
    expect(result.num_nested_selections).toBe(0);
  });
});
