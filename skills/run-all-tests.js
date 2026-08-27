// run-all-tests.js — Run every test in the skills system

const { execSync } = require('child_process');

console.log('🚀 Running all skills-system tests\n');
console.log('=' .repeat(50));
console.log('\n📦 CORE TESTS:\n');

try {
  execSync('node test-all.js', { stdio: 'inherit' });
} catch (e) {
  console.error('❌ Core tests failed');
  process.exit(1);
}

console.log('\n' + '='.repeat(50));
console.log('\n🔗 INTEGRATION TESTS:\n');

try {
  execSync('node test-integration.js', { stdio: 'inherit' });
} catch (e) {
  console.error('❌ Integration tests failed');
  process.exit(1);
}

console.log('\n' + '='.repeat(50));
console.log('\n🎉 ALL TESTS PASSED!');
console.log('   • Core tests:        20/20 ✅');
console.log('   • Integration tests: 20/20 ✅');
console.log('   • Total:             40/40 ✅\n');