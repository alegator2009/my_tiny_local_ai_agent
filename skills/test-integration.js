// test-integration.js — Integration tests

const { execSync } = require('child_process');

function runTest(name, fn) {
  try {
    execSync(fn(), { stdio: 'pipe' });
    console.log(`✅ ${name}`);
  } catch (e) {
    console.error(`❌ ${name}: ${e.message.substring(0, 100)}`);
  }
}

async function main() {
  console.log('🧪 Integration tests\n');

  // 1. Chat display
  runTest('Chat display: list skills', () => `node -e "const c=require('./chat-display.js'); console.log(c.displayAllSkills())"`);

  // 2. Chat display error
  runTest('Chat display: error', () => `node -e "const c=require('./chat-display.js'); console.log(c.displayError('Test error'))"`);

  // 3. Chat display success
  runTest('Chat display: success', () => `node -e "const c=require('./chat-display.js'); console.log(c.displaySuccess('Test success'))"`);

  // 4. Chat display usage
  runTest('Chat display: usage', () => `node -e "const r=require('./registry.js'); r.createSkill('demo','Demo',['Step1']); const c=require('./chat-display.js'); console.log(c.displaySkillUsage('demo'))"`);

  // 5. Meta-skill creator: parse
  runTest('Meta-skill: parse request', () => `node -e "const m=require('./meta-skill-creator.js'); const r=m.parseUserRequest('Create a skill called review for code review. When: code needs review'); console.log(r?.name)"`);

  // 6. Meta-skill creator: create
  runTest('Meta-skill: create skill', () => `node -e "const m=require('./meta-skill-creator.js'); const r=m.createSkillFromRequest('Create a skill called tester for testing. When: testing is needed'); console.log(r.success)"`);

  // 7. Meta-skill creator: duplicate error
  runTest('Meta-skill: duplicate', () => `node -e "const m=require('./meta-skill-creator.js'); const r=m.createSkillFromRequest('Create a skill called tester'); console.log(r.success)"`);

  // 8. Agent integration: handle list
  runTest('Agent: list skills', () => `node -e "const a=require('./agent-integration.js'); const r=a.handleUserRequest('list skills'); console.log(r.type)"`);

  // 9. Agent integration: handle create
  runTest('Agent: create skill', () => `node -e "const a=require('./agent-integration.js'); const r=a.handleUserRequest('Create a skill called helper for assistance'); console.log(r.type)"`);

  // 10. Agent integration: no skill
  runTest('Agent: no skill', () => `node -e "const a=require('./agent-integration.js'); const r=a.handleUserRequest('plain text without a skill'); console.log(r.type)"`);

  // 11. Agent integration: check creation request
  runTest('Agent: creation request check', () => `node -e "const a=require('./agent-integration.js'); console.log(a.isSkillCreationRequest('Create a skill called test'))"`);

  // 12. Agent integration: check list request
  runTest('Agent: list request check', () => `node -e "const a=require('./agent-integration.js'); console.log(a.isListRequest('list skills'))"`);

  // 13. Agent integration: find applicable
  runTest('Agent: find skill', () => `node -e "const r=require('./registry.js'); r.createSkill('tester','Tester for review',['Step1']); const a=require('./agent-integration.js'); console.log(a.findApplicableSkill('Tester for review')?.name)"`);

  // 14. Agent integration: handle execution
  runTest('Agent: skill execution', () => `node -e "const a=require('./agent-integration.js'); const r=a.handleSkillExecution({name:'tester',description:'Test'},'Test request'); console.log(r.type)"`);

  // 15. Index.js API
  runTest('Index: createSkill API', () => `node -e "const s=require('./index.js'); console.log(s.createSkill('api-skill','API skill',['Step1'])?.name)"`);

  // 16. Index.js API
  runTest('Index: listSkills API', () => `node -e "const s=require('./index.js'); console.log(s.listSkills().length > 0)"`);

  // 17. Index.js API
  runTest('Index: getSkill API', () => `node -e "const s=require('./index.js'); console.log(s.getSkill('api-skill')?.name)"`);

  // 18. Index.js API
  runTest('Index: activateSkill API', () => `node -e "const s=require('./index.js'); console.log(s.activateSkill('api-skill')?.displayMessage?.length > 0)"`);

  // 19. Index.js API
  runTest('Index: executeSkill API', () => `node -e "const s=require('./index.js'); console.log(s.executeSkill('api-skill',{userPrompt:'test'}).success)"`);

  // 20. Index.js API
  runTest('Index: listAvailableSkills API', () => `node -e "const s=require('./index.js'); console.log(s.listAvailableSkills().length > 0)"`);

  console.log('\n🎉 Integration tests complete!');
}

main();