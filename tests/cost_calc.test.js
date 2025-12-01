// tests/cost_calc.test.js

import { describe, test, expect } from "vitest";
import { parse, buildSchema } from "graphql";
import { computeCostMetrics } from "../src/ingestion/parser/cost_calc.js";


// ------------------------------------------------------
// Minimal schema for testing
// ------------------------------------------------------
const schema = buildSchema(`
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
`);


// ------------------------------------------------------
// Tests
// ------------------------------------------------------

describe("Cost Calculator - Phase 1 (Complete Coverage)", () => {

  // ------------------------------------------------------
  test("computes cost for a simple query", async () => {
    const query = `
      query {
        simple
      }
    `;

    const ast = parse(query);
    const result = await computeCostMetrics({ schema, ast });

    expect(result.error).toBe(null);
    expect(result.estimated_cost).toBeGreaterThan(0);
    expect(result.complexity_score).toBe(result.estimated_cost);
  });

  // ------------------------------------------------------
  test("nested selections increase cost", async () => {
    const query = `
      query {
        user(id: 1) {
          details {
            email
            address
          }
        }
      }
    `;

    const ast = parse(query);
    const result = await computeCostMetrics({ schema, ast });

    // cost ~ number of fields visited
    expect(result.error).toBe(null);
    expect(result.estimated_cost).toBeGreaterThan(3);
  });

  // ------------------------------------------------------
  test("arguments do not break cost evaluation", async () => {
    const query = `
      query {
        posts(limit: 10, offset: 5) {
          title
        }
      }
    `;

    const ast = parse(query);
    const result = await computeCostMetrics({ schema, ast });

    expect(result.error).toBe(null);
    expect(result.estimated_cost).toBeGreaterThan(1);
  });

  // ------------------------------------------------------
  test("fragment-based queries compute cost correctly", async () => {
    const query = `
      query {
        user(id: 1) {
          ...UserFields
        }
      }

      fragment UserFields on User {
        id
        name
      }
    `;

    const ast = parse(query);
    const result = await computeCostMetrics({ schema, ast });

    expect(result.error).toBe(null);
    expect(result.estimated_cost).toBeGreaterThanOrEqual(2);
  });

  // ------------------------------------------------------
  test("invalid query returns error", async () => {
    const badQuery = `
      query {
        invalidField
      }
    `;

    const ast = parse(badQuery);
    const result = await computeCostMetrics({ schema, ast });

    expect(result.error).not.toBe(null);
    expect(result.estimated_cost).toBe(null);
  });

  // ------------------------------------------------------
  test("missing schema returns fallback info", async () => {
    const query = `
      query {
        simple
      }
    `;

    const ast = parse(query);
    const result = await computeCostMetrics({ schema: null, ast });

    expect(result.error).toContain("Schema not provided");
    expect(result.estimated_cost).toBe(null);
    expect(result.complexity_score).toBe(null);
  });

  // ------------------------------------------------------
  test("multiple operations compute cost normally", async () => {
    const query = `
      query A {
        simple
      }

      query B {
        user(id: 1) { id }
      }
    `;

    const ast = parse(query);
    const result = await computeCostMetrics({ schema, ast });

    expect(result.error).toBe(null);
    expect(result.estimated_cost).toBeGreaterThan(1);
  });

});
