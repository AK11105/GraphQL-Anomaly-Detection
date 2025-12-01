// src/ingestion/parser/ast_extractor.js

import { parse, visit } from "graphql";

/**
 * Parse a raw GraphQL query and extract structural AST metadata.
 * This is Phase-1 only: AST + simple counts, no cost/depth here.
 */

export function extractAST(queryString) {
  const result = {
    ast: null,
    nodes: [],
    stats: {
      num_fields: 0,
      num_fragments: 0,
      num_directives: 0,
      num_aliases: 0,
      num_operations: 0,
      num_mutations: 0,
      num_subscriptions: 0,
      operation_type: null,
      num_variables: 0,
      num_introspection_operations: 0,
      num_arguments: 0,
    },
    error: null,
  };

  if (!queryString || typeof queryString !== "string") {
    result.error = "Invalid or empty query string.";
    return result;
  }

  // -----------------------------
  // 1. Parse GraphQL query → AST
  // -----------------------------
  try {
    result.ast = parse(queryString);
    //console.log(JSON.stringify(result.ast, null, 2)); // Add this line
  } catch (err) {
    result.error = `AST parsing failed: ${err.message}`;
    return result;
  }

  // -----------------------------
  // 2. Walk AST → collect metadata
  // -----------------------------
  visit(result.ast, {
    OperationDefinition(node) {
      result.stats.num_operations += 1;

      if (node.operation === "query") {
        result.stats.operation_type = "query";
      } else if (node.operation === "mutation") {
        result.stats.num_mutations += 1;
        result.stats.operation_type = "mutation";
      } else if (node.operation === "subscription") {
        result.stats.num_subscriptions += 1;
        result.stats.operation_type = "subscription";
      }

      if (node.variableDefinitions) {
        result.stats.num_variables += node.variableDefinitions.length;
      }
    },

    Field(node) {
      //console.log('Field:', node.name?.value);
      result.stats.num_fields += 1;

      if (node.alias) {
        result.stats.num_aliases += 1;
      }

      if (node.arguments && node.arguments.length > 0) {
        result.stats.num_arguments += node.arguments.length;
      }

      // Introspection detection
      const fieldName = node.name?.value;
      if (fieldName && fieldName.startsWith("__")) {
        result.stats.num_introspection_operations += 1;
      }

      // Save lightweight node metadata
      result.nodes.push({
        kind: "Field",
        name: fieldName || null,
        alias: node.alias?.value || null,
        argCount: node.arguments?.length || 0,
        selectionCount: node.selectionSet?.selections?.length || 0,
      });
    },

    FragmentDefinition(node) {
      result.stats.num_fragments += 1;
      result.nodes.push({
        kind: "FragmentDefinition",
        name: node.name?.value || null,
      });
    },

    FragmentSpread(node) {
      result.stats.num_fragments += 1;
      result.nodes.push({
        kind: "FragmentSpread",
        name: node.name?.value || null,
      });
    },

    Directive(node) {
      result.stats.num_directives += 1;
      result.nodes.push({
        kind: "Directive",
        name: node.name?.value || null,
      });
    },
  });

  return result;
}
