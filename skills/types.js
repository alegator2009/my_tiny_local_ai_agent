// types.js — JSDoc typedefs for the skills system
/**
 * @typedef {Object} Skill
 * @property {string} name - Skill name
 * @property {string} description - Short description
 * @property {string[]} instructions - Step-by-step instructions
 * @property {string} [whenToUse] - When to use this skill
 * @property {{prompt: string, action: string}[]} [examples] - Usage examples
 * @property {Object} [delegates_to] - Optional MCP delegation descriptor
 * @property {Object} [tool_args]    - Optional literal tool arguments
 * @property {string} createdAt - Creation timestamp (ISO)
 * @property {string} updatedAt - Last update timestamp (ISO)
 */

/**
 * @typedef {Object} SkillRegistry
 * @property {Record<string, import('./types').Skill>} skills
 * @property {number} version
 * @property {string} lastModified
 */

/**
 * @typedef {Object} Observation
 * @property {string} kind       - "user" | "tool" | "model" | "system" | "note"
 * @property {string|null} source
 * @property {string} content
 * @property {Object} [meta]
 * @property {string} timestamp
 */

/**
 * @typedef {Object} SkillState
 * @property {string} skillName
 * @property {number} schemaVersion
 * @property {"running"|"completed"|"failed"} status
 * @property {number} currentStep     - 1-based pointer into instructions
 * @property {number} totalSteps
 * @property {Object} variables        - Skill-written key/value bag
 * @property {Observation[]} history   - Bounded ring buffer
 * @property {Observation|null} lastObservation
 * @property {Object|null} pendingTransition
 * @property {string|null} error
 * @property {number} maxHistory
 * @property {number} iterations
 * @property {string} createdAt
 * @property {string} updatedAt
 */

/**
 * @typedef {Object} Transition
 * @property {"advance"|"set-variable"|"complete"|"fail"|"retry"} kind
 * @property {Object} [set]      - Variable assignments (set-variable).
 * @property {string} [error]    - Failure reason (fail).
 */

/**
 * @typedef {Object} PromptInputs
 * @property {Skill} spec
 * @property {Object} state
 * @property {Observation|null} observation
 * @property {Observation[]} history
 */