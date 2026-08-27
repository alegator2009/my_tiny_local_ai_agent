// registry.js — Skill registry (storage, creation, lookup)

const fs = require('node:fs');
const path = require('node:path');

const REGISTRY_PATH = path.join(__dirname, 'registry.json');

/**
 * Default registry contents.
 */
const DEFAULT_REGISTRY = {
  skills: {},
  version: 1,
  lastModified: new Date().toISOString(),
};

/**
 * Load the skill registry from disk.
 */
function loadRegistry() {
  try {
    if (fs.existsSync(REGISTRY_PATH)) {
      const content = fs.readFileSync(REGISTRY_PATH, 'utf8');
      return JSON.parse(content);
    }
  } catch (error) {
    console.error('[Skill Registry] Failed to load registry:', error.message);
  }

  saveRegistry(DEFAULT_REGISTRY);
  return DEFAULT_REGISTRY;
}

/**
 * Persist the skill registry to disk.
 */
function saveRegistry(registry) {
  registry.lastModified = new Date().toISOString();
  fs.writeFileSync(REGISTRY_PATH, JSON.stringify(registry, null, 2), 'utf8');
}

/**
 * Create a new skill and persist it.
 * @param {string} name
 * @param {string} description
 * @param {string[]} instructions
 * @param {string} [whenToUse]
 * @param {Array<{prompt: string, action: string}>} [examples]
 */
function createSkill(name, description, instructions, whenToUse, examples) {
  const now = new Date().toISOString();

  const skill = {
    name,
    description,
    instructions: instructions || [],
    createdAt: now,
    updatedAt: now,
  };

  if (whenToUse) skill.whenToUse = whenToUse;

  if (examples && Array.isArray(examples) && examples.length > 0) {
    skill.examples = examples.map((ex) => ({
      prompt: ex.prompt || '',
      action: ex.action || '',
    }));
  }

  const registry = loadRegistry();

  if (registry.skills[name]) {
    console.warn(`[Skill Registry] Skill "${name}" already exists. Overwriting.`);
  }

  registry.skills[name] = skill;
  saveRegistry(registry);

  return skill;
}

/**
 * Return the list of all registered skills.
 */
function listSkills() {
  const registry = loadRegistry();
  return Object.values(registry.skills);
}

/**
 * Look up a skill by name.
 */
function getSkill(name) {
  const registry = loadRegistry();
  return registry.skills[name];
}

/**
 * Remove a skill from the registry.
 */
function deleteSkill(name) {
  const registry = loadRegistry();

  if (!registry.skills[name]) {
    console.warn(`[Skill Registry] Skill "${name}" not found.`);
    return false;
  }

  delete registry.skills[name];
  saveRegistry(registry);
  return true;
}

/**
 * Update an existing skill.
 */
function updateSkill(name, updates) {
  const registry = loadRegistry();

  if (!registry.skills[name]) {
    console.warn(`[Skill Registry] Skill "${name}" not found.`);
    return undefined;
  }

  registry.skills[name].updatedAt = new Date().toISOString();
  Object.assign(registry.skills[name], updates);
  saveRegistry(registry);

  return registry.skills[name];
}

module.exports = { loadRegistry, saveRegistry, createSkill, listSkills, getSkill, deleteSkill, updateSkill };