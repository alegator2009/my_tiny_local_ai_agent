// Focused regression test for the validated persistence used by skill-author.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-infinite-skills-'));
process.env.SKILLS_REGISTRY_PATH = path.join(tempDir, 'registry.json');

const { getSkill, saveSkillDefinition } = require('./registry');

const definition = {
  name: 'migration-review',
  description: 'Review database migrations before deployment.',
  whenToUse: 'When a user asks to review a database migration.',
  instructions: ['Inspect the migration.', 'Flag irreversible or unsafe changes.'],
};

const created = saveSkillDefinition(definition);
assert.equal(created.success, true);
assert.equal(created.action, 'created');
assert.equal(getSkill('migration-review').description, definition.description);

const duplicate = saveSkillDefinition(definition);
assert.equal(duplicate.success, false);

const updated = saveSkillDefinition(
  { ...definition, description: 'Review database migrations before a production deployment.' },
  { overwrite: true },
);
assert.equal(updated.success, true);
assert.equal(updated.action, 'updated');
assert.equal(getSkill('migration-review').admin, undefined);

fs.rmSync(tempDir, { recursive: true, force: true });
console.log('skill-author registry validation passed');
