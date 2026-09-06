const readline = require('readline');
const { defaultAgent } = require('./agent/core');
const { executeTool } = require('./agent/tools');

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  red: '\x1b[31m',
  bgDark: '\x1b[40m'
};

async function startCli() {
  console.clear();
  console.log(`${colors.cyan}${colors.bright}====================================================${colors.reset}`);
  console.log(`${colors.cyan}${colors.bright}              ✨ SOUL AI - LOCAL AI ASSISTANT       ${colors.reset}`);
  console.log(`${colors.dim}       100% Offline • Private • NVIDIA RTX 4060       ${colors.reset}`);
  console.log(`${colors.cyan}${colors.bright}====================================================${colors.reset}\n`);

  // Check health
  const health = await defaultAgent.checkHealth();
  if (health.online) {
    console.log(`${colors.green}● Local Engine Status: ONLINE${colors.reset} (${health.endpoint})`);
    console.log(`${colors.blue}  Active Model: ${health.activeModel}${colors.reset}`);
    if (health.models.length > 0) {
      console.log(`${colors.dim}  Available Models: ${health.models.join(', ')}${colors.reset}`);
    }
    // Pre-load the model into VRAM so the first reply doesn't time out on cold load
    defaultAgent.warmup().then(warmed => {
      if (warmed) {
        console.log(`${colors.green}  Engine warmed & ready (fast first response)${colors.reset}`);
      } else {
        console.log(`${colors.yellow}  Model warmup skipped${colors.reset}`);
      }
    }).catch(() => {});
  } else {
    console.log(`${colors.yellow}● Local Engine Status: STANDBY (Ollama not detected)${colors.reset}`);
    console.log(`${colors.dim}  Run: powershell .\\setup-ollama.ps1 to install & pull Llama/Qwen/Mistral/Gemma${colors.reset}`);
    console.log(`${colors.dim}  Fallback simulation mode active.${colors.reset}`);
  }

  console.log(`\n${colors.dim}Type your request, 'help' for commands, or 'exit' to quit.${colors.reset}\n`);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `${colors.magenta}${colors.bright}User > ${colors.reset}`
  });

  const conversation = [];

  rl.prompt();

  rl.on('line', async (line) => {
    const input = line.trim();
    if (!input) {
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'exit' || input.toLowerCase() === 'quit') {
      console.log(`\n${colors.cyan}Goodbye! Offline Agent terminated.${colors.reset}`);
      process.exit(0);
    }

    if (input.toLowerCase() === 'clear') {
      console.clear();
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'models') {
      const h = await defaultAgent.checkHealth();
      console.log(`\nAvailable local models: ${h.models.join(', ') || 'None found. Pull with: ollama pull qwen2.5:7b'}\n`);
      rl.prompt();
      return;
    }

    if (input.toLowerCase().startsWith('use ')) {
      const newModel = input.slice(4).trim();
      defaultAgent.setOptions({ model: newModel });
      console.log(`\nSwitched active model to: ${colors.green}${newModel}${colors.reset}\n`);
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'sys') {
      const info = await executeTool('getSystemInfo', {});
      console.log(`\n${colors.yellow}=== System Telemetry ===${colors.reset}`);
      console.log(JSON.stringify(info, null, 2));
      console.log('');
      rl.prompt();
      return;
    }

    conversation.push({ role: 'user', content: input });

    process.stdout.write(`\n${colors.blue}${colors.bright}Agent Thinking...${colors.reset}\n`);

    try {
      await defaultAgent.runChat(conversation, {
        onThought: (thought) => {
          console.log(`${colors.dim}💭 [Thought] ${thought}${colors.reset}`);
        },
        onToolCall: (call) => {
          console.log(`${colors.yellow}⚡ [Tool Executing] ${call.name}(${JSON.stringify(call.arguments || {})})${colors.reset}`);
        },
        onToolResult: (res) => {
          const str = JSON.stringify(res.result);
          console.log(`${colors.green}✔ [Tool Result] ${str.length > 150 ? str.slice(0, 150) + '...' : str}${colors.reset}`);
        },
        onToken: (token) => {
          process.stdout.write(token);
        }
      });
      console.log('\n');
    } catch (err) {
      console.log(`\n${colors.red}Error: ${err.message}${colors.reset}\n`);
    }

    rl.prompt();
  });
}

startCli();
