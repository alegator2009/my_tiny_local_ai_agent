// executor.js — SKILL.state executor
//
// Implements the runtime loop from arXiv:2608.26263. The executor never
// touches a conversation history. Each invocation sees:
//
//   * spec  : the immutable skill definition (from registry.js)
//   * state : the current SkillState (built by state.js)
//   * observation : the latest observation, produced by the caller
//                   (a tool result, the user's request, etc.)
//
// The executor's job is to:
//   1. Build the model inputs from (spec, state, observation).
//   2. Run a model call (skipped when no model is wired in — see
//      `callModel` below; tests and the local-only path can run the
//      executor in `dry-run` mode).
//   3. Validate the model's structured transition.
//   4. Apply it to the state and append the resulting observation.
//   5. Return a serialisable `StepResult` for the chat/UI.
//
// Intermediate reasoning produced by the model is discarded — only the
// validated state update is kept.

'use strict';

const {
  createInitialState,
  pushObservation,
  validateTransition,
  applyTransition,
  buildPromptInputs,
  resetState,
  TransitionError,
  MAX_HISTORY_DEFAULT,
} = require('./state');

const { getSkill } = require('./registry');

/**
 * Optional model hook. When the surrounding runtime wants to actually
 * invoke an LLM, it can inject a `callModel` factory:
 *
 *   setModelCaller(async (promptInputs) => ({
 *     reasoning: '...',     // discarded after validation
 *     transition: { kind: 'advance' },
 *     observation: { kind: 'note', content: '...' },
 *   }))
 *
 * In the absence of a model caller the executor runs in `dry-run`
 * mode, advancing the state machine one step at a time without
 * producing free-form text — this is what `wrapper.mjs` and the
 * integration tests rely on.
 */
let _modelCaller = null;
function setModelCaller(fn) {
  _modelCaller = typeof fn === 'function' ? fn : null;
}
function getModelCaller() {
  return _modelCaller;
}

/**
 * Run a single SKILL.state step.
 *
 * @param {string} name
 * @param {object} state       SkillState (will be mutated in place).
 * @param {object} [opts]
 * @param {object} [opts.observation]  Optional observation to push
 *                                     before running the step.
 * @param {string} [opts.kind]         Transition kind (default "advance").
 * @returns {{success: boolean, message: string, nextAction?: string,
 *           state: object, promptInputs: object,
 *           observation?: object}}
 */
function executeStep(name, state, opts = {}) {
  if (!state || typeof state !== 'object') {
    return {
      success: false,
      message: `❌ SkillState required to execute "${name}".`,
      state: null,
      promptInputs: null,
    };
  }
  const skill = getSkill(name);
  if (!skill) {
    return {
      success: false,
      message: `❌ Skill "${name}" not found.`,
      state,
      promptInputs: null,
    };
  }
  if (state.skillName !== name) {
    return {
      success: false,
      message: `❌ State belongs to skill "${state.skillName}", not "${name}".`,
      state,
      promptInputs: null,
    };
  }
  if (state.status === 'completed') {
    return {
      success: true,
      message: `✅ Skill "${name}" already completed (step ${state.currentStep}/${state.totalSteps}).`,
      state,
      promptInputs: buildPromptInputs(state),
    };
  }
  if (state.status === 'failed') {
    return {
      success: false,
      message: `❌ Skill "${name}" previously failed: ${state.error || 'unknown'}. Pass kind="retry" to resume.`,
      state,
      promptInputs: buildPromptInputs(state),
    };
  }

  if (opts.observation) {
    pushObservation(state, opts.observation);
  }

  const promptInputs = buildPromptInputs(state);

  // If a model is wired in, consult it. Otherwise we advance
  // deterministically (dry-run / tests).
  const caller = getModelCaller();
  if (caller) {
    return _runWithModel(name, state, promptInputs, opts);
  }
  return _runDry(name, state, promptInputs, opts);
}

function _runDry(name, state, promptInputs, opts) {
  const kind = opts.kind || 'advance';
  const observation = opts.observationResult || null;
  try {
    applyTransition(state, { kind }, observation);
  } catch (err) {
    if (err instanceof TransitionError) {
      return {
        success: false,
        message: `❌ Transition rejected for "${name}": ${err.message}`,
        state,
        promptInputs,
        error: { code: err.code, message: err.message },
      };
    }
    throw err;
  }
  return _buildStepResult(name, state, promptInputs, observation);
}

async function _runWithModel(name, state, promptInputs, opts) {
  const caller = getModelCaller();
  let modelOut;
  try {
    modelOut = await caller(promptInputs, opts);
  } catch (err) {
    applyTransition(state, { kind: 'fail', error: `model_error: ${err.message}` });
    return {
      success: false,
      message: `❌ Model call failed for "${name}": ${err.message}`,
      state,
      promptInputs,
      error: { code: 'model_error', message: String(err.message || err) },
    };
  }
  const transition = (modelOut && modelOut.transition) || { kind: opts.kind || 'advance' };
  const observation = modelOut && modelOut.observation
    ? modelOut.observation
    : opts.observationResult || null;
  try {
    applyTransition(state, transition, observation);
  } catch (err) {
    if (err instanceof TransitionError) {
      return {
        success: false,
        message: `❌ Model produced invalid transition for "${name}": ${err.message}`,
        state,
        promptInputs,
        error: { code: err.code, message: err.message },
      };
    }
    throw err;
  }
  // Discard reasoning immediately per the SKILL.state contract.
  return _buildStepResult(name, state, promptInputs, observation);
}

function _buildStepResult(name, state, promptInputs, observation) {
  const skill = getSkill(name);
  const stepIndex = Math.max(0, state.currentStep - 1);
  const instruction = skill.instructions[stepIndex] || '(done)';
  let message;
  if (state.status === 'completed') {
    message = `✅ Skill "${name}" completed after ${state.iterations} iteration(s).`;
  } else if (state.status === 'failed') {
    message = `❌ Skill "${name}" failed: ${state.error || 'unknown'}.`;
  } else {
    message = `📝 Step ${state.currentStep}/${state.totalSteps}: ${instruction}`;
  }
  const result = {
    success: state.status !== 'failed',
    message,
    state,
    promptInputs,
  };
  if (observation) {
    result.observation = observation;
  }
  if (state.status === 'running') {
    result.nextAction = `Continue with step ${state.currentStep + 1}`;
  }
  return result;
}

/**
 * Legacy-style helper retained for backward compatibility. Builds a
 * fresh state from the user's prompt, runs one dry step, and returns
 * the legacy `{success, message, nextAction, context}` shape used by
 * older callers plus the new `state`/`promptInputs` fields.
 */
function executeSkill(name, context) {
  const initial = createInitialState(name, {
    userPrompt: context && context.userPrompt,
    maxHistory: context && context.maxHistory,
    variables: context && context.variables,
  });
  if (context && context.variables) {
    Object.assign(initial.variables, context.variables);
  }
  const observation = context && context.observation
    ? context.observation
    : (context && context.userPrompt
      ? { kind: 'user', source: 'user', content: context.userPrompt }
      : null);
  const result = executeStep(name, initial, {
    observation,
    kind: 'advance',
  });
  return {
    ...result,
    nextAction: result.nextAction,
    context: {
      ...(context || {}),
      currentStep: initial.currentStep,
      variables: initial.variables,
      state: initial,
    },
  };
}

function executeSkillWithDisplay(name, context) {
  const skill = getSkill(name);
  if (!skill) {
    return {
      result: { success: false, message: `❌ Skill "${name}" not found.` },
      displayMessage: '',
    };
  }
  const displayMessage = formatSkillForDisplay(skill);
  const result = executeSkill(name, context);
  return { result, displayMessage };
}

function formatSkillForDisplay(skill) {
  let message = `📋 **Activated skill:** ${skill.name}\n\n`;
  if (skill.description) {
    message += `**Description:** ${skill.description}\n\n`;
  }
  if (skill.whenToUse) {
    message += `**When to use:** ${skill.whenToUse}\n\n`;
  }
  return message;
}

function isSkillApplicable(skillName, userPrompt) {
  const skill = getSkill(skillName);
  if (!skill) return false;

  const lowerPrompt = (userPrompt || '').toLowerCase();
  const lowerDescription = (skill.description || '').toLowerCase();
  const lowerWhenToUse = (skill.whenToUse || '').toLowerCase();

  const promptWords = new Set(lowerPrompt.split(/\s+/));
  const descriptionWords = new Set(lowerDescription.split(/\s+/));
  const whenToUseWords = new Set(lowerWhenToUse.split(/\s+/));

  let matchCount = 0;
  for (const word of promptWords) {
    if (descriptionWords.has(word) || whenToUseWords.has(word)) {
      matchCount++;
    }
  }
  return matchCount >= 2;
}

/**
 * Run a skill to completion synchronously in dry-run mode. Useful for
 * deterministic tests and for the MCP wrapper when no model is wired
 * in. Each step records an observation built from `observations[i]` if
 * provided, otherwise from the previous step's output.
 */
function runSkillDry(name, opts = {}) {
  const observations = Array.isArray(opts.observations) ? opts.observations : [];
  const state = createInitialState(name, {
    userPrompt: opts.userPrompt,
    maxHistory: opts.maxHistory,
    variables: opts.variables,
  });
  const trace = [];
  let observation = observations[0] || (opts.userPrompt
    ? { kind: 'user', source: 'user', content: opts.userPrompt }
    : null);
  let safety = (opts.maxIterations || (state.totalSteps + observations.length + 4));
  while (state.status === 'running' && safety-- > 0) {
    const stepResult = executeStep(name, state, { observation, kind: 'advance' });
    trace.push(stepResult);
    if (!stepResult.success) break;
    if (state.status !== 'running') break;
    const next = observations[trace.length] || null;
    observation = next;
  }
  return { state, trace };
}

module.exports = {
  executeStep,
  executeSkill,
  executeSkillWithDisplay,
  formatSkillForDisplay,
  isSkillApplicable,
  runSkillDry,
  setModelCaller,
  getModelCaller,
  createInitialState,
  resetState,
  MAX_HISTORY_DEFAULT,
};