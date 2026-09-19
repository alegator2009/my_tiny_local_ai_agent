// registry.js — Skill registry (storage, creation, lookup)

const fs = require('node:fs');
const path = require('node:path');

const REGISTRY_PATH = process.env.SKILLS_REGISTRY_PATH || path.join(__dirname, 'registry.json');

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
function createSkill(name, description, instructions, whenToUse, examples, options = {}) {
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
  if (options.delegates_to && typeof options.delegates_to === 'object') {
    skill.delegates_to = options.delegates_to;
  }
  if (options.tool_args && typeof options.tool_args === 'object') {
    skill.tool_args = options.tool_args;
  }
  if (options.admin && typeof options.admin === 'string') {
    skill.admin = options.admin;
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

/**
 * Validate the portable JSON shape used by the skill-author MCP tool.
 * Admin-only fields deliberately cannot be imported: only the built-in
 * skill-author definition may grant registry-write capabilities.
 */
function normalizeSkillDefinition(definition) {
  if (!definition || typeof definition !== 'object' || Array.isArray(definition)) {
    return { error: 'definition must be an object' };
  }
  const name = String(definition.name || '').trim().toLowerCase();
  if (!/^[a-z0-9][a-z0-9-]{0,62}$/.test(name)) {
    return { error: 'name must use lowercase letters, digits, and hyphens (1–63 characters)' };
  }
  const description = String(definition.description || '').trim();
  if (!description) return { error: 'description is required' };
  const instructions = Array.isArray(definition.instructions)
    ? definition.instructions.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 24)
    : [];
  if (instructions.length === 0) return { error: 'at least one instruction is required' };
  const normalized = {
    name,
    description,
    instructions,
  };
  const whenToUse = String(definition.whenToUse || '').trim();
  if (whenToUse) normalized.whenToUse = whenToUse;
  if (Array.isArray(definition.examples)) {
    normalized.examples = definition.examples.slice(0, 8)
      .filter((item) => item && typeof item === 'object')
      .map((item) => ({ prompt: String(item.prompt || ''), action: String(item.action || '') }));
  }
  if (definition.delegates_to && typeof definition.delegates_to === 'object') {
    const tool = String(definition.delegates_to.tool || '').trim();
    if (!tool) return { error: 'delegates_to.tool is required when delegates_to is supplied' };
    normalized.delegates_to = {
      tool,
      args_from: Array.isArray(definition.delegates_to.args_from)
        ? definition.delegates_to.args_from.map(String).slice(0, 12)
        : [],
      default_args: definition.delegates_to.default_args && typeof definition.delegates_to.default_args === 'object'
        ? definition.delegates_to.default_args
        : {},
    };
  }
  return { definition: normalized };
}

/** Create a new definition or explicitly update an existing one. */
function saveSkillDefinition(definition, { overwrite = false } = {}) {
  const parsed = normalizeSkillDefinition(definition);
  if (parsed.error) return { success: false, message: parsed.error };
  const next = parsed.definition;
  const existing = getSkill(next.name);
  if (existing && !overwrite) {
    return { success: false, message: `Skill "${next.name}" already exists. Use action "update" to replace it.` };
  }
  if (existing) {
    const updated = updateSkill(next.name, {
      description: next.description,
      instructions: next.instructions,
      ...(next.whenToUse ? { whenToUse: next.whenToUse } : {}),
      ...(next.examples ? { examples: next.examples } : {}),
      ...(next.delegates_to ? { delegates_to: next.delegates_to } : {}),
    });
    return { success: true, action: 'updated', skill: updated };
  }
  const created = createSkill(
    next.name,
    next.description,
    next.instructions,
    next.whenToUse,
    next.examples,
    { delegates_to: next.delegates_to },
  );
  return { success: true, action: 'created', skill: created };
}

module.exports = {
  loadRegistry,
  saveRegistry,
  createSkill,
  listSkills,
  getSkill,
  deleteSkill,
  updateSkill,
  normalizeSkillDefinition,
  saveSkillDefinition,
};
