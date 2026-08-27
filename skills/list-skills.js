#!/usr/bin/env node
/**
 * list-skills.js — prints the list of all skills from registry.json.
 * Usage: node list-skills.js
 *        node list-skills.js --json
 */
const path = require("path");
const { listSkills } = require("./registry.js");

const args = process.argv.slice(2);
const asJson = args.includes("--json");

if (asJson) {
  const all = listSkills();
  process.stdout.write(JSON.stringify(all, null, 2) + "\n");
  process.exit(0);
}

const all = listSkills();
const created = all
  .map((s) => ({
    name: s.name,
    description: s.description || "(no description)",
    instructions: (s.instructions || []).length,
    examples: (s.examples || []).length,
  }))
  .sort((a, b) => a.name.localeCompare(b.name));

console.log(`\n📚 Skills (${all.length}):\n`);
for (const s of created) {
  console.log(`  • ${s.name}`);
  console.log(`     description: ${s.description}`);
  console.log(`     steps:       ${s.instructions}`);
  console.log(`     examples:    ${s.examples}`);
}
console.log();