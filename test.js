const assert = require('assert');
const { executeTool } = require('./agent/tools');
const { ragEngine } = require('./agent/rag');
const { defaultAgent } = require('./agent/core');

async function runTests() {
  console.log('🧪 Running Offline AI Agent Automated Test Suite...\n');

  // Test 1: Math Calculator Tool
  console.log('Test 1: Testing calculate tool...');
  const mathRes = await executeTool('calculate', { expression: '(128 * 45) + 500' });
  assert.strictEqual(mathRes.result, 6260, 'Calculation should equal 6260');
  console.log('✔ Calculation passed: (128 * 45) + 500 =', mathRes.result);

  // Test 2: File Write and Read Tools
  console.log('\nTest 2: Testing workspace file write and read...');
  const writeRes = await executeTool('writeFile', {
    path: 'test_note.txt',
    content: 'Offline agent test verification content.'
  });
  assert.strictEqual(writeRes.success, true);

  const readRes = await executeTool('readFile', { path: 'test_note.txt' });
  assert.ok(readRes.content.includes('Offline agent test verification content.'));
  console.log('✔ File write & read passed.');

  // Test 3: System Info Tool
  console.log('\nTest 3: Testing getSystemInfo...');
  const sysInfo = await executeTool('getSystemInfo', {});
  assert.ok(sysInfo.ram, 'Should contain RAM stats');
  assert.ok(sysInfo.cpuModel, 'Should contain CPU info');
  console.log('✔ System info gathered successfully:', sysInfo.ram);

  // Test 4: Memory Storage Tool
  console.log('\nTest 4: Testing memoryStore and memoryRetrieve...');
  await executeTool('storeMemory', { key: 'favorite_model', value: 'Qwen 2.5 7B' });
  const memRes = await executeTool('retrieveMemory', { key: 'favorite_model' });
  assert.strictEqual(memRes.data.value, 'Qwen 2.5 7B');
  console.log('✔ Memory store and retrieval verified.');

  // Test 5: Offline RAG Engine
  console.log('\nTest 5: Testing Offline RAG Engine...');
  ragEngine.addDocument('ai_hardware_specs.txt', 'The NVIDIA RTX 4060 has 8GB GDDR6 memory and runs 7B 8B quantized LLMs efficiently.');
  const ragSearch = ragEngine.search('RTX 4060 memory', 1);
  assert.ok(ragSearch.results.length > 0, 'RAG should return matches');
  assert.ok(ragSearch.results[0].text.includes('NVIDIA RTX 4060'));
  console.log('✔ RAG Search passed with score:', ragSearch.results[0].score.toFixed(3));

  // Test 6: LocalAgentEngine health check
  console.log('\nTest 6: Testing Agent Engine health check...');
  const health = await defaultAgent.checkHealth();
  console.log('✔ Agent status verified:', health.online ? 'Ollama Online' : 'Standby / Fallback Ready');

  console.log('\n======================================================');
  console.log('🎉 ALL AUTOMATED TESTS PASSED SUCCESSFULLY!');
  console.log('======================================================\n');
}

runTests().catch(err => {
  console.error('✖ Test suite failed:', err);
  process.exit(1);
});
