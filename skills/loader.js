// loader.js — Skill loader and activator (SKILL.state aware)

const { getSkill, listSkills } = require('./registry');
const {
  createInitialState,
  buildPromptInputs,
} = require('./state');

/**
 * Activate a skill and return its (spec, state, observation) triple —
 * exactly the runtime inputs the SKILL.state paper feeds to the model.
 *
 * @param {string} name
 * @param {object} [opts]
 * @param {string} [opts.userPrompt]
 * @param {object} [opts.variables]
 * @param {number} [opts.maxHistory]
 * @returns {{skill: object, displayMessage: string,
 *           state: object, promptInputs: object} | null}
 */
function activateSkill(name, opts = {}) {
  const skill = getSkill(name);
  if (!skill) return null;

  const displayMessage = formatSkillDisplay(skill);
  const state = createInitialState(name, {
    userPrompt: opts.userPrompt,
    variables: opts.variables,
    maxHistory: opts.maxHistory,
  });
  const promptInputs = buildPromptInputs(state);

  return { skill, displayMessage, state, promptInputs };
}

function formatSkillDisplay(skill) {
  let message = `📋 **Activated skill:** ${skill.name}\n\n`;
  if (skill.description) {
    message += `**Description:** ${skill.description}\n\n`;
  }
  if (skill.whenToUse) {
    message += `**When to use:** ${skill.whenToUse}\n\n`;
  }
  if (skill.instructions && skill.instructions.length > 0) {
    message += `**Instructions:**\n`;
    skill.instructions.forEach((inst, index) => {
      message += `${index + 1}. ${inst}\n`;
    });
    message += '\n';
  }
  if (skill.examples && skill.examples.length > 0) {
    message += `**Examples:**\n`;
    skill.examples.forEach((ex, index) => {
      message += `### Example ${index + 1}\n`;
      message += `- **Request:** ${ex.prompt}\n`;
      message += `- **Action:** ${ex.action}\n\n`;
    });
  }
  return message;
}

function listAvailableSkills() {
  const skills = listSkills();
  if (skills.length === 0) {
    return '📋 **Available skills:**\n\n❌ No skills found.\n\nUse `createSkill` to register a new one.';
  }
  let message = '📋 **Available skills:**\n\n';
  skills.forEach((skill) => {
    message += `- **${skill.name}**: ${skill.description}\n`;
  });
  return message;
}

function skillExists(name) {
  const skill = getSkill(name);
  return skill !== undefined;
}

/**
 * Serialize an entire prompt bundle (spec + state + observation) to a
 * markdown rendering suitable for human inspection in the chat UI or
 * for logging in the transcript.
 */
function renderPromptBundle(promptInputs) {
  if (!promptInputs) return '';
  const { spec, state, observation, history = [] } = promptInputs;
  let out = `### SKILL.state prompt bundle\n\n`;
  out += `**Spec — ${spec.name}**\n${spec.description || ''}\n\n`;
  if (spec.whenToUse) {
    out += `*When to use:* ${spec.whenToUse}\n\n`;
  }
  out += `**Instructions (${spec.instructions.length}):**\n`;
  spec.instructions.forEach((inst, idx) => {
    const pointer = idx + 1 === state.currentStep ? '▶️' : '·';
    out += `${pointer} ${idx + 1}. ${inst}\n`;
  });
  out += `\n**Execution state:**\n\`\`\`json\n${JSON.stringify(state, null, 2)}\n\`\`\`\n`;
  if (observation) {
    out += `\n**Latest observation (${observation.kind}${observation.source ? ' / ' + observation.source : ''}):**\n${observation.content}\n`;
  }
  if (history.length > 0) {
    out += `\n**History (${history.length}/${state.variables && state.variables.maxHistory ? state.variables.maxHistory : '?'}):**\n`;
    history.forEach((o, idx) => {
      out += `${idx + 1}. [${o.kind}] ${o.content.slice(0, 200)}\n`;
    });
  }
  return out;
}

module.exports = {
  activateSkill,
  formatSkillDisplay,
  listAvailableSkills,
  skillExists,
  renderPromptBundle,
};