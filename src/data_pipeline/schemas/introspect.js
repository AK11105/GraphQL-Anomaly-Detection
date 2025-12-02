// introspect.js
import fs from "fs";
import { buildSchema, graphql, getIntrospectionQuery } from "graphql";

const sdlFile = process.argv[2] || "schema_sdl.graphql";
const outFile = process.argv[3] || "schema.json";

const sdl = fs.readFileSync(sdlFile, "utf8");
const schema = buildSchema(sdl);

const result = await graphql({
  schema,
  source: getIntrospectionQuery(),
});

if (result.errors) {
  console.error(result.errors);
  process.exit(1);
}

fs.writeFileSync(outFile, JSON.stringify(result.data, null, 2));
console.log("Wrote", outFile);
