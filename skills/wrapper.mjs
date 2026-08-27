// wrapper.mjs — MCP server that exposes registered skills as tools.
//
// Implements the SKILL.state contract (arXiv:2608.26263). Every skill
// tool call accepts an optional `state` field; the wrapper loads /
// validates that state, executes one validated step, and returns the
// new state alongside the chat-friendly message. Reasoning produced
// by the model is never persisted.
//
// Skill kinds supported:
//
//   1. Self-contained — wrapper.mjs returns the SKILL.state bundle
//      for the model to act on, then applies a caller-supplied
//      transition.
//   2. Delegating (skill has a `delegates_to` field) — wrapper.mjs
//      returns a `DELEGATE:...` directive for the API.
//   3. Tool-args based (skill has a `tool_args` field) — wrapper.mjs
//      returns the literal `TOOL_ARGS:...` envelope.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REGISTRY_PATH = path.join(__dirname, 'registry.json');

function loadSkills() {
  if (!fs.existsSync(REGISTRY_PATH)) return [];
  const registry = JSON.parse(fs.readFileSync(REGISTRY_PATH, 'utf8'));
  return Object.values(registry.skills || {});
}

function send(message) {
  process.stdout.write(JSON.stringify(message) + '\n');
}

function readRequest() {
  return new Promise((resolve) => {
    let buffer = '';
    const onData = (chunk) => {
      buffer += chunk.toString('utf8');
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) {
          try {
            const msg = JSON.parse(trimmed);
            resolve(msg);
            return;
          } catch {
            // Ignore invalid JSON lines.
          }
        }
      }
    };
    process.stdin.on('data', onData);
    process.stdin.on('end', () => resolve(null));
  });
}

/**
 * Build the inputSchema for a SKILL.state tool. Every tool accepts:
 *   * `state`       — optional SkillState (the model passes back what
 *                     it received last call).
 *   * `transition`  — optional structured transition to apply.
 *   * `observation` — optional observation to push before running.
 *
 * For delegating skills we additionally surface the delegated tool's
 * argument names so the model can pass them through.
 */
function buildInputSchema(skill) {
  const properties = {
    state: {
      type: 'object',
      description:
        'SkillState returned by the previous call. Omit to start a fresh state.',
      additionalProperties: true,
    },
    transition: {
      type: 'object',
      description:
        'Structured SKILL.state transition. Kind: advance | set-variable | complete | fail | retry.',
      properties: {
        kind: { type: 'string', enum: ['advance', 'set-variable', 'complete', 'fail', 'retry'] },
        set: { type: 'object', additionalProperties: true },
        error: { type: 'string' },
      },
    },
    observation: {
      type: 'object',
      description: 'Observation to push before applying the transition.',
      properties: {
        kind: { type: 'string' },
        source: { type: 'string' },
        content: { type: 'string' },
        meta: { type: 'object', additionalProperties: true },
      },
    },
    args: {
      type: 'object',
      description: 'Free-form skill arguments (echoed back in the bundle).',
      additionalProperties: true,
    },
  };

  if (skill.delegates_to && skill.delegates_to.args_from) {
    for (const argName of skill.delegates_to.args_from) {
      properties[argName] = {
        type: 'string',
        description: `Argument "${argName}" for the delegated tool`,
      };
    }
  }

  return {
    type: 'object',
    properties,
    additionalProperties: true,
  };
}

/**
 * Pure helper that derives an `initial SkillState` for a skill.
 * Mirrors `state.createInitialState` from state.js but is reproduced
 * here so the wrapper stays self-contained for `node`/`mcp` hosts
 * that do not bundle the rest of the skills/ tree.
 */
function buildInitialState(skill, args) {
  const totalSteps = Array.isArray(skill.instructions) ? skill.instructions.length : 0;
  const userPrompt =
    args && typeof args === 'object' && typeof args.observation?.content === 'string'
      ? args.observation.content
      : (args && typeof args === 'object' && typeof args.userPrompt === 'string'
          ? args.userPrompt
          : null);

  return {
    skillName: skill.name,
    schemaVersion: 1,
    status: 'running',
    currentStep: 1,
    totalSteps,
    variables: {},
    history: [],
    lastObservation: null,
    pendingTransition: null,
    error: null,
    maxHistory: 6,
    iterations: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    userPrompt,
  };
}

function coerceState(skill, args) {
  if (args && typeof args === 'object' && args.state && typeof args.state === 'object') {
    // Carry the supplied state forward, but always re-anchor the
    // skillName / totalSteps from the registry so the model can't drift.
    return {
      ...args.state,
      skillName: skill.name,
      totalSteps: Array.isArray(skill.instructions) ? skill.instructions.length : 0,
      updatedAt: new Date().toISOString(),
    };
  }
  return buildInitialState(skill, args);
}

/**
 * Validate a transition against the current state. Returns either
 * `{ ok: true, state }` or `{ ok: false, error }`.
 */
function applyTransitionPure(state, transition) {
  const t = transition && typeof transition === 'object' ? transition : { kind: 'advance' };
  const kind = t.kind || 'advance';
  const next = {
    ...state,
    variables: { ...state.variables },
    history: state.history.slice(),
    meta: { ...(state.meta || {}) },
  };
  next.iterations = (state.iterations || 0) + 1;
  next.pendingTransition = { ...t, validatedAt: new Date().toISOString() };

  switch (kind) {
    case 'advance': {
      if (state.status !== 'running') {
        return { ok: false, error: `cannot advance from status "${state.status}"` };
      }
      if (state.currentStep >= state.totalSteps) {
        next.status = 'completed';
        next.currentStep = state.totalSteps;
      } else {
        next.currentStep = state.currentStep + 1;
      }
      break;
    }
    case 'set-variable': {
      if (!t.set || typeof t.set !== 'object') {
        return { ok: false, error: 'set-variable requires `set` object' };
      }
      for (const [k, v] of Object.entries(t.set)) {
        next.variables[k] = v;
      }
      break;
    }
    case 'complete':
      next.status = 'completed';
      next.error = null;
      break;
    case 'fail':
      next.status = 'failed';
      next.error = t.error ? String(t.error) : 'unspecified';
      break;
    case 'retry':
      if (state.status !== 'failed') {
        return { ok: false, error: 'retry requires failed state' };
      }
      next.status = 'running';
      next.error = null;
      break;
    default:
      return { ok: false, error: `unknown transition kind "${kind}"` };
  }

  if (!['running', 'completed', 'failed'].includes(next.status)) {
    return { ok: false, error: `invalid status "${next.status}"` };
  }
  next.updatedAt = new Date().toISOString();
  return { ok: true, state: next };
}

function pushObservationPure(state, observation) {
  if (!observation || typeof observation !== 'object') return state;
  const record = {
    kind: String(observation.kind || 'note'),
    source: observation.source ? String(observation.source) : null,
    content: observation.content == null ? '' : String(observation.content),
    meta: observation.meta && typeof observation.meta === 'object' ? { ...observation.meta } : {},
    timestamp: new Date().toISOString(),
  };
  const history = state.history.slice();
  history.push(record);
  while (history.length > state.maxHistory) {
    history.shift();
  }
  return { ...state, history, lastObservation: record };
}

/**
 * Render the SKILL.state prompt bundle for the model.
 */
function buildPromptBundle(skill, state) {
  return {
    spec: {
      name: skill.name,
      description: skill.description || '',
      whenToUse: skill.whenToUse || '',
      instructions: Array.isArray(skill.instructions) ? skill.instructions.slice() : [],
      examples: Array.isArray(skill.examples) ? skill.examples.slice() : [],
    },
    state: {
      skillName: state.skillName,
      status: state.status,
      currentStep: state.currentStep,
      totalSteps: state.totalSteps,
      variables: { ...state.variables },
      iterations: state.iterations || 0,
      error: state.error || null,
    },
    observation: state.lastObservation
      ? {
          kind: state.lastObservation.kind,
          source: state.lastObservation.source,
          content: state.lastObservation.content,
          meta: { ...(state.lastObservation.meta || {}) },
        }
      : null,
    history: state.history.map((o) => ({
      kind: o.kind,
      source: o.source,
      content: o.content,
      timestamp: o.timestamp,
    })),
  };
}

async function executeSkill(skill, args) {
  // --- DELEGATING SKILL ---
  if (skill.delegates_to) {
    const delegate = {
      delegate: true,
      tool: skill.delegates_to.tool,
      args: {},
    };
    if (Array.isArray(skill.delegates_to.args_from)) {
      for (const argName of skill.delegates_to.args_from) {
        if (args && Object.prototype.hasOwnProperty.call(args, argName)) {
          delegate.args[argName] = args[argName];
        }
      }
    }
    if (skill.delegates_to.args_map) {
      for (const [src, dst] of Object.entries(skill.delegates_to.args_map)) {
        if (args && Object.prototype.hasOwnProperty.call(args, src)) {
          delegate.args[dst] = args[src];
        }
      }
    }
    if (Object.keys(delegate.args).length === 0) {
      if (skill.delegates_to.default_args && typeof skill.delegates_to.default_args === 'object') {
        delegate.args = { ...skill.delegates_to.default_args };
      }
      if (Object.keys(delegate.args).length === 0 && args && typeof args === 'object') {
        if (args.args && typeof args.args === 'object') {
          delegate.args = { ...args.args };
        } else {
          // Filter out SKILL.state envelope fields before forwarding.
          const filtered = { ...args };
          for (const k of ['state', 'transition', 'observation']) {
            delete filtered[k];
          }
          delegate.args = filtered;
        }
      }
    }

    // Even for delegating skills we still update the state so the
    // caller has a SKILL.state-shaped response. The reasoning is
    // discarded after validation per the paper.
    const state = coerceState(skill, args);
    const withObservation = args && args.observation
      ? pushObservationPure(state, args.observation)
      : state;
    const validated = applyTransitionPure(withObservation, { kind: 'advance' });
    const finalState = validated.ok ? validated.state : state;
    return {
      content: [
        {
          type: 'text',
          text:
            'DELEGATE:' +
            JSON.stringify(delegate) +
            '\n\nSKILL_STATE:' +
            JSON.stringify(buildPromptBundle(skill, finalState)),
        },
      ],
      isError: false,
    };
  }

  // --- TOOL-ARGS BASED ---
  if (skill.tool_args) {
    const state = coerceState(skill, args);
    return {
      content: [
        {
          type: 'text',
          text:
            'TOOL_ARGS:' +
            JSON.stringify(skill.tool_args) +
            '\n\nSKILL_STATE:' +
            JSON.stringify(buildPromptBundle(skill, state)),
        },
      ],
    };
  }

  // --- SELF-CONTAINED (default) ---
  let state = coerceState(skill, args);
  if (args && args.observation) {
    state = pushObservationPure(state, args.observation);
  }
  if (args && args.transition) {
    const result = applyTransitionPure(state, args.transition);
    if (!result.ok) {
      return {
        content: [
          {
            type: 'text',
            text:
              '❌ Invalid transition: ' +
              result.error +
              '\n\nSKILL_STATE:' +
              JSON.stringify(buildPromptBundle(skill, state)),
          },
        ],
        isError: true,
      };
    }
    state = result.state;
  } else {
    // No transition supplied — return the bundle for the model to act on.
    state = applyTransitionPure(state, { kind: 'advance' }).state || state;
  }
  const bundle = buildPromptBundle(skill, state);

  const instructionsText = (skill.instructions || [])
    .map((i, idx) => `${idx + 1}. ${i}`)
    .join('\n');

  const header = `🔧 Using skill: ${skill.name}\n📝 ${skill.description}\n`;
  const result = `${header}\nInstructions:\n${instructionsText}\n\nReceived arguments:\n${JSON.stringify(args, null, 2)}\n\nSKILL_STATE bundle:\n${JSON.stringify(bundle, null, 2)}`;
  return {
    content: [{ type: 'text', text: result }],
  };
}

async function handleRequest(req) {
  if (!req || !req.method) return;

  if (req.method === 'initialize') {
    send({
      jsonrpc: '2.0',
      id: req.id,
      result: {
        protocolVersion: '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: { name: 'skills-mcp', version: '2.0.0-skillstate' },
      },
    });
    return;
  }

  if (req.method === 'notifications/initialized') {
    return;
  }

  if (req.method === 'tools/list') {
    const skills = loadSkills();
    const tools = skills.map((skill) => ({
      name: skill.name,
      description: (skill.description || `Run skill: ${skill.name}`) + ' [SKILL.state]',
      inputSchema: buildInputSchema(skill),
    }));
    send({ jsonrpc: '2.0', id: req.id, result: { tools } });
    return;
  }

  if (req.method === 'tools/call') {
    const { name, arguments: args = {} } = req.params || {};
    const skills = loadSkills();
    const skill = skills.find((s) => s.name === name);
    if (!skill) {
      send({
        jsonrpc: '2.0',
        id: req.id,
        error: { code: -32602, message: `Skill "${name}" not found` },
      });
      return;
    }
    const result = await executeSkill(skill, args);
    send({ jsonrpc: '2.0', id: req.id, result });
    return;
  }
}

async function main() {
  send({
    jsonrpc: '2.0',
    method: 'notifications/message',
    params: {
      level: 'info',
      data: 'Skills MCP started (v2.0.0-skillstate — SKILL.state runtime)',
    },
  });

  while (true) {
    const req = await readRequest();
    if (!req) break;
    await handleRequest(req);
  }
}

main().catch((e) => {
  process.stderr.write(`[skills-mcp] fatal: ${e.message}\n`);
  process.exit(1);
});