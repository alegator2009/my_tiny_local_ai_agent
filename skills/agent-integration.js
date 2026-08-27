// agent-integration.js — Skill system integration for the AI agent
//
// Routes user requests through the SKILL.state runtime:
//   * createSkill / listSkills still live in the registry / loader.
//   * executeSkill now goes through `executeStep` so the model sees
//     only (spec, state, observation) and intermediate reasoning is
//     discarded.

const { listSkills, getSkill } = require('./registry');
const {
  executeStep,
  executeSkill,
  isSkillApplicable,
} = require('./executor');
const { createInitialState } = require('./state');
const { createSkillFromRequest } = require('./meta-skill-creator');
const {
  displaySkillUsage,
  displaySkillActivation,
  displayAllSkills,
  displayError,
  displaySuccess,
} = require('./chat-display');

function handleUserRequest(userPrompt) {
  if (isSkillCreationRequest(userPrompt)) {
    return handleSkillCreation(userPrompt);
  }

  if (isListRequest(userPrompt)) {
    return {
      type: 'skills-list',
      message: displayAllSkills(),
    };
  }

  const applicableSkill = findApplicableSkill(userPrompt);

  if (applicableSkill) {
    return handleSkillExecution(applicableSkill, userPrompt);
  }

  return {
    type: 'no-skill',
    message: '🤔 No matching skill found. Here is what is available:\n' + displayAllSkills(),
  };
}

function isSkillCreationRequest(prompt) {
  if (!prompt) return false;
  const lower = prompt.toLowerCase();
  return /create\s+skill|new\s+skill|build\s+skill/i.test(lower);
}

function isListRequest(prompt) {
  if (!prompt) return false;
  const lower = prompt.toLowerCase();
  return /list\s+skills|available\s+skills|show\s+skills/i.test(lower);
}

function findApplicableSkill(userPrompt) {
  const skills = listSkills();
  for (const skill of skills) {
    if (isSkillApplicable(skill.name, userPrompt)) {
      return skill;
    }
  }
  return null;
}

function handleSkillCreation(userPrompt) {
  const result = createSkillFromRequest(userPrompt);
  if (!result.success) {
    return {
      type: 'skill-creation-error',
      message: displayError(result.message),
    };
  }
  return {
    type: 'skill-created',
    message: result.message,
    skill: result.skill,
  };
}

/**
 * SKILL.state-aware execution path: build a fresh state, run one
 * validated step, and return both the chat banner and the structured
 * state so the orchestrator can serialize it.
 */
function handleSkillExecution(skill, userPrompt) {
  const activationMessage = displaySkillActivation(skill.name);
  const state = createInitialState(skill.name, { userPrompt });
  const observation = userPrompt
    ? { kind: 'user', source: 'user', content: userPrompt }
    : null;
  const result = executeStep(skill.name, state, {
    observation,
    kind: 'advance',
  });

  return {
    type: 'skill-execution',
    message:
      activationMessage +
      '\n' +
      (result.message || displaySuccess('Skill executed')),
    skill,
    result,
    state,
  };
}

// Backward-compatible legacy helper that preserves the original
// behaviour of returning a flat `{success, message, nextAction, context}`.
function legacyExecuteSkill(name, context) {
  return executeSkill(name, context);
}

module.exports = {
  handleUserRequest,
  isSkillCreationRequest,
  isListRequest,
  findApplicableSkill,
  handleSkillCreation,
  handleSkillExecution,
  legacyExecuteSkill,
};