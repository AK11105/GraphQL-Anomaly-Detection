// tests/ast_extractor.test.js

import { describe, test, expect } from "vitest";
import { extractAST } from "../src/ingestion/parser/ast_extractor.js";

// --------------------
// Test Queries
// --------------------

const QUERY_BASIC = `
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      profile: details {
        name
        email
      }
      posts(limit: 5, offset: 10) {
        title
      }
    }
  }
`;

const QUERY_WITH_FRAGMENTS = `
  query GetUser {
    user {
      ...UserFields
    }
  }

  fragment UserFields on User {
    id
    email
  }
`;

const QUERY_WITH_DIRECTIVES = `
  query GetStuff($active: Boolean!) {
    items @include(if: $active) @deprecated {
      id
      name
      meta(limit: 10) @skip(if: false)
    }
  }
`;

const MUTATION_AND_SUBSCRIPTION = `
  mutation UpdateUser {
    updateUser(id: 1) { id }
  }

  subscription UserUpdates {
    userUpdated { id }
  }
`;

const QUERY_WITH_INTROSPECTION = `
  {
    __schema {
      types {
        name
      }
    }
    __typename
  }
`;

// -------------------------------------------------------
// Tests
// -------------------------------------------------------

describe("AST Extractor - Phase 1 (Complete Coverage)", () => {

  // -------------------------------------------------------
  test("parses basic query and extracts correct metadata", () => {
    const result = extractAST(QUERY_BASIC);

    expect(result.error).toBe(null);
    expect(result.ast).not.toBe(null);

    expect(result.stats.num_operations).toBe(1);
    expect(result.stats.operation_type).toBe("query");
    expect(result.stats.num_variables).toBe(1);

    // Fields detected correctly
    expect(result.stats.num_fields).toBe(7); 
    // user, id, details(profile alias), name, email, posts, title

    expect(result.stats.num_aliases).toBe(1);  // profile: details
    expect(result.stats.num_arguments).toBe(3); // id, limit, offset

    expect(result.stats.num_fragments).toBe(0);
    expect(result.stats.num_directives).toBe(0);
    expect(result.stats.num_introspection_operations).toBe(0);

    expect(result.nodes.length).toBeGreaterThan(5);
  });

  // -------------------------------------------------------
  test("extracts fragments and fragment spreads correctly", () => {
    const result = extractAST(QUERY_WITH_FRAGMENTS);

    expect(result.error).toBe(null);

    // fragment definition + fragment spread = 2
    expect(result.stats.num_fragments).toBe(2);

    const fragmentNodes = result.nodes.filter(n =>
      n.kind === "FragmentDefinition" || n.kind === "FragmentSpread"
    );
    expect(fragmentNodes.length).toBe(2);
  });

  // -------------------------------------------------------
  test("extracts directives, arguments, and nested field metadata", () => {
    const result = extractAST(QUERY_WITH_DIRECTIVES);

    expect(result.error).toBe(null);

    expect(result.stats.num_directives).toBe(3); // @include, @deprecated, @skip
    expect(result.stats.num_arguments).toBe(1);  // meta(limit: 10)

    // fields inside item + meta + id + name
    expect(result.stats.num_fields).toBeGreaterThanOrEqual(4);
  });

  // -------------------------------------------------------
  test("detects mutations and subscriptions correctly", () => {
    const result = extractAST(MUTATION_AND_SUBSCRIPTION);

    expect(result.error).toBe(null);

    expect(result.stats.num_operations).toBe(2);
    expect(result.stats.num_mutations).toBe(1);
    expect(result.stats.num_subscriptions).toBe(1);

    // operation_type stores the *last* operation visited
    expect(["mutation", "subscription"]).toContain(result.stats.operation_type);
  });

  // -------------------------------------------------------
  test("detects introspection fields", () => {
    const result = extractAST(QUERY_WITH_INTROSPECTION);

    expect(result.error).toBe(null);
    expect(result.stats.num_introspection_operations).toBe(2);
    // __schema and __typename → 2
  });

  // -------------------------------------------------------
  test("returns error on invalid query", () => {
    const result = extractAST("query { invalid(");
    expect(result.error).not.toBe(null);
    expect(result.ast).toBe(null);
  });

});
