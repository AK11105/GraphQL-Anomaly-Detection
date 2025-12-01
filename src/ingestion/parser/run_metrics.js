// src/ingestion/parser/run_metrics.js

// Disable all logs — Python requires pure JSON
console.log = () => {};
console.error = () => {};
console.warn = () => {};

import fs from "fs";
import { extractAST } from "./ast_extractor.js";
import { computeDepthMetrics } from "./depth_calc.js";
import { computeCostMetrics } from "./cost_calc.js";

const stdinBuffer = fs.readFileSync(0, "utf-8");

let payload = null;
try {
  payload = JSON.parse(stdinBuffer);
} catch (err) {
  process.stdout.write(JSON.stringify({ error: "Invalid input JSON" }));
  process.exit(0);
}

const { query, schemaSDL } = payload;

let schema = null;

if (schemaSDL) {
  try {
    const { buildSchema } = await import("graphql");
    schema = buildSchema(schemaSDL);
  } catch (err) {
    // Schema remains null
  }
}

const astResult = extractAST(query);

let depth = null;
let cost = null;

if (astResult.ast) {
  depth = computeDepthMetrics(astResult.ast);
  cost = await computeCostMetrics({ schema, ast: astResult.ast });
}

const output = {
  ast: astResult,
  depth,
  cost,
};

process.stdout.write(JSON.stringify(output));
