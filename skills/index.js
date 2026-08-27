// index.js — Public API for the skills system (SKILL.state runtime)

const registry = require('./registry');
const loader = require('./loader');
const executor = require('./executor');
const state = require('./state');

// Registry shortcuts
function createSkill(name, description, instructions, whenToUse) {
  return registry.createSkill(name, description, instructions, whenToUse);
}
function listSkills() { return registry.listSkills(); }
function getSkill(name) { return registry.getSkill(name); }
function deleteSkill(name) { return registry.deleteSkill(name); }

// Activation now returns the full SKILL.state bundle.
function activateSkill(name, opts = {}) {
  return loader.activateSkill(name, opts);
}

// New canonical step-based executor entry point.
function executeStep(name, stateObj, opts = {}) {
  return executor.executeStep(name, stateObj, opts);
}

// Legacy entry point retained for backward compatibility.
function executeSkill(name, context) {
  return executor.executeSkill(name, context);
}

function executeSkillWithDisplay(name, context) {
  return executor.executeSkillWithDisplay(name, context);
}

function isSkillApplicable(skillName, userPrompt) {
  return executor.isSkillApplicable(skillName, userPrompt);
}

function formatSkillDisplay(name, opts) {
  const activation = loader.activateSkill(name, opts);
  return activation ? activation.displayMessage : '';
}

function listAvailableSkills() {
  return loader.listAvailableSkills();
}

function renderPromptBundle(promptInputs) {
  return loader.renderPromptBundle(promptInputs);
}

function runSkillDry(name, opts = {}) {
  return executor.runSkillDry(name, opts);
}

function setModelCaller(fn) {
  executor.setModelCaller(fn);
}

module.exports = {
  // registry
  createSkill,
  listSkills,
  getSkill,
  deleteSkill,
  // activation / display
  activateSkill,
  formatSkillDisplay,
  listAvailableSkills,
  renderPromptBundle,
  // execution
  executeStep,
  executeSkill,
  executeSkillWithDisplay,
  isSkillApplicable,
  runSkillDry,
  setModelCaller,
  // state
  createInitialState: state.createInitialState,
  validateTransition: state.validateTransition,
  applyTransition: state.applyTransition,
  pushObservation: state.pushObservation,
  buildPromptInputs: state.buildPromptInputs,
  resetState: state.resetState,
  TransitionError: state.TransitionError,
  MAX_HISTORY_DEFAULT: state.MAX_HISTORY_DEFAULT,
};