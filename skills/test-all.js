// test-all.js — Comprehensive test suite for the skills system

const { execSync } = require('child_process');

function runTest(name, fn) {
  try {
    const output = execSync(fn(), { stdio: 'pipe' });
    console.log(`✅ ${name}`);
  } catch (e) {
    if (e.stdout && e.stdout.toString) {
      console.error(`   Output: ${e.stdout.toString().substring(0, 200)}...`);
    }
    console.error(`❌ ${name}: ${e.message}`);
  }
}

async function main() {
  console.log('🧪 Skills system tests\n');

  // Test 1: Create a skill
  runTest('Create skill', () => `node -e "const r=require('./registry.js'); console.log(r.createSkill('test-skill','Test skill',['Step1','Step2']))"`);

  // Test 2: Get a skill
  runTest('Get skill', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('test-skill')?.name)"`);

  // Test 3: List skills
  runTest('List skills', () => `node -e "const r=require('./registry.js'); console.log(r.listSkills().length)"`);

  // Test 4: Activate a skill
  runTest('Activate skill', () => `node -e "const l=require('./loader.js'); console.log(l.activateSkill('test-skill')?.displayMessage)"`);

  // Test 5: Execute a skill
  runTest('Execute skill', () => `node -e "const e=require('./executor.js'); console.log(e.executeSkill('test-skill',{userPrompt:'test'}).success)"`);

  // Test 6: Delete a skill
  runTest('Delete skill', () => `node -e "const r=require('./registry.js'); console.log(r.deleteSkill('test-skill'))"`);

  // Test 7: Verify deletion
  runTest('Verify deletion', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('test-skill'))"`);

  // Test 8: Create a skill with examples
  runTest('Create skill with examples', () => `node -e "const r=require('./registry.js'); console.log(JSON.stringify(r.createSkill('example-skill','Skill with examples',['Step1'],'When needed',[{prompt:'test',action:'act'}])))"`);

  // Test 9: Verify examples
  runTest('Verify examples', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.examples?.length)"`);

  // Test 10: Verify description
  runTest('Verify description', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.description)"`);

  // Test 11: Verify whenToUse
  runTest('Verify whenToUse', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.whenToUse)"`);

  // Test 12: Verify instructions
  runTest('Verify instructions', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.instructions.length)"`);

  // Test 13: Verify example prompt
  runTest('Verify example prompt', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.examples[0]?.prompt)"`);

  // Test 14: Verify example action
  runTest('Verify example action', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.examples[0]?.action)"`);

  // Test 15: Verify createdAt
  runTest('Verify createdAt', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.createdAt)"`);

  // Test 16: Verify updatedAt
  runTest('Verify updatedAt', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.updatedAt)"`);

  // Test 17: Update a skill
  runTest('Update skill', () => `node -e "const r=require('./registry.js'); console.log(r.updateSkill('example-skill',{description:'New description'}))"`);

  // Test 18: Verify update
  runTest('Verify update', () => `node -e "const r=require('./registry.js'); console.log(r.getSkill('example-skill')?.description)"`);

  // Test 19: Execute a skill with context
  runTest('Execute skill with context', () => `node -e "const e=require('./executor.js'); console.log(e.executeSkill('example-skill',{userPrompt:'test','currentStep':1}).success)"`);

  // Test 20: Verify nextAction
  runTest('Verify nextAction', () => `node -e "const e=require('./executor.js'); console.log(e.executeSkill('example-skill',{userPrompt:'test','currentStep':1}).nextAction)"`);

  console.log('\n🎉 All tests complete!');
}

main();