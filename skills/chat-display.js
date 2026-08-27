// chat-display.js — Skill presentation helpers for the chat surface

const { getSkill, listSkills } = require('./registry');
const { activateSkill } = require('./loader');

/**
 * Render a "skill in use" banner.
 */
function displaySkillUsage(skillName) {
  const skill = getSkill(skillName);
  if (!skill) return '';

  let message = `\n🔧 **Using skill:** ${skill.name}\n`;
  if (skill.description) {
    message += `📝 ${skill.description}\n`;
  }
  return message;
}

/**
 * Render a skill-activation banner.
 */
function displaySkillActivation(skillName) {
  const activation = activateSkill(skillName);
  if (!activation) {
    return `❌ Skill "${skillName}" not found.`;
  }
  return `\n📋 **Activated skill:** ${activation.skill.name}\n\n${activation.displayMessage}\n`;
}

/**
 * Render the list of available skills.
 */
function displayAllSkills() {
  const skills = listSkills();
  if (skills.length === 0) {
    return '\n📚 **Available skills:**\n\n❌ No skills found.\n';
  }

  let message = '\n📚 **Available skills:**\n\n';
  skills.forEach((skill) => {
    message += `🔹 **${skill.name}** — ${skill.description}\n`;
  });
  return message;
}

/**
 * Render a banner after a new skill is created.
 */
function displaySkillCreated(skill) {
  return `\n✨ **New skill created:** ${skill.name}\n📝 ${skill.description}\n`;
}

/**
 * Render a banner after a skill is removed.
 */
function displaySkillDeleted(skillName) {
  return `\n🗑️ **Skill removed:** ${skillName}\n`;
}

/**
 * Render an error banner.
 */
function displayError(message) {
  return `\n❌ **Error:** ${message}\n`;
}

/**
 * Render a success banner.
 */
function displaySuccess(message) {
  return `\n✅ **Success:** ${message}\n`;
}

module.exports = {
  displaySkillUsage,
  displaySkillActivation,
  displayAllSkills,
  displaySkillCreated,
  displaySkillDeleted,
  displayError,
  displaySuccess,
};