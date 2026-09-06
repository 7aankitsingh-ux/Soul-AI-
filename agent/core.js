const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');
const { tools, executeTool, getToolDefinitions } = require('./tools');
const { buildSystemPrompt } = require('./prompts');

const QWEN_TEACHER_JSONL = path.join(__dirname, '..', 'soul-llm', 'data', 'qwen_teacher.jsonl');
const SOUL_LLM_API = 'http://localhost:8000';

class LocalAgentEngine {
  constructor(options = {}) {
    this.provider = options.provider || 'ollama'; // 'ollama' | 'openai-compatible' | 'demo'
    this.endpoint = options.endpoint || 'http://localhost:11434';
    this.model = options.model || 'qwen2.5:7b';
    this.temperature = options.temperature !== undefined ? options.temperature : 0.7;
    this.customInstructions = options.customInstructions || '';
    this.maxIterations = options.maxIterations || 6;
    // Distillation collection mode (Qwen = teacher): 'auto' | 'manual' | 'after-n'
    //   auto   : every successful Qwen reply is saved to the teacher dataset
    //   manual : nothing saved automatically (use saveManualExample)
    //   after-n: auto-save AND ask the training worker to start when `retrainAfter`
    //            new examples are pending
    this.collectMode = options.collectMode || 'auto';
    this.retrainAfter = Number(options.retrainAfter) || 10;
    // Preferred engine (explicit user choice). SOUL-LLM is only an emergency fallback.
    this.preferred = {
      provider: this.provider,
      endpoint: this.endpoint,
      model: this.model
    };
  }

  setOptions(newOpts = {}) {
    if (newOpts.provider) { this.provider = newOpts.provider; this.preferred.provider = newOpts.provider; }
    if (newOpts.endpoint) { this.endpoint = newOpts.endpoint; this.preferred.endpoint = newOpts.endpoint; }
    if (newOpts.model) { this.model = newOpts.model; this.preferred.model = newOpts.model; }
    if (newOpts.temperature !== undefined) this.temperature = Number(newOpts.temperature);
    if (newOpts.customInstructions !== undefined) this.customInstructions = newOpts.customInstructions;
    if (newOpts.maxIterations !== undefined) this.maxIterations = Number(newOpts.maxIterations);
    if (newOpts.collectMode !== undefined) this.collectMode = String(newOpts.collectMode);
    if (newOpts.retrainAfter !== undefined) this.retrainAfter = Number(newOpts.retrainAfter) || 10;
  }

  // ─── Qwen teacher -> SOUL-LLM training data pipeline ──────────────
  // Every helper below is best-effort: a capture/notify failure must NEVER
  // break the ongoing chat.

  // Persist a teacher example: {prompt, response, timestamp, model}
  async saveTeacherExample(prompt, response) {
    if (!prompt || typeof response !== 'string' || !response.trim()) return false;
    try {
      const dir = path.dirname(QWEN_TEACHER_JSONL);
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(QWEN_TEACHER_JSONL,
        JSON.stringify({
          prompt,
          response: response.trim(),
          timestamp: new Date().toISOString(),
          model: this.model
        }) + '\n', 'utf8');
      return true;
    } catch (err) {
      console.error('[SOUL-LLM] Failed to save teacher example:', err.message);
      return false;
    }
  }

  // Auto-collect every successful chat reply (unless mode is 'manual')
  async captureConversation(messages, response) {
    if (this.collectMode === 'manual') return;
    const userMsg = (messages || []).filter(m => m.role === 'user').pop()?.content || '';
    if (!userMsg) return;

    const saved = await this.saveTeacherExample(userMsg, response);
    if (!saved) return;

    const count = this.datasetSize();
    console.log(`[SOUL-LLM] Teacher example captured (#${count}) — Qwen → ${this.model}`);

    // 'after-n' mode as well as plain 'auto' rely on the background worker to
    // batch-train. Notify it now (idempotent — it only trains when enough
    // new examples are pending).
    await this.notifyTrainer();
  }

  // Manual mode: the UI calls this to save one specific Q&A pair.
  async saveManualExample(prompt, response) {
    const saved = await this.saveTeacherExample(prompt, response);
    if (saved) console.log(`[SOUL-LLM] Manually saved teacher example (${this.datasetSize()} total)`);
    return saved;
  }

  datasetSize() {
    try {
      if (!fs.existsSync(QWEN_TEACHER_JSONL)) return 0;
      return fs.readFileSync(QWEN_TEACHER_JSONL, 'utf8').split('\n').filter(l => l.trim()).length;
    } catch {
      return 0;
    }
  }

  // Ensure the SOUL-LLM training worker is running and configured.
  async notifyTrainer() {
    try {
      // Keep the worker's batching threshold in sync with the UI setting.
      await this.makeRequest(SOUL_LLM_API + '/api/training/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ train_after: this.retrainAfter })
      }, 5000);
      await this.makeRequest(SOUL_LLM_API + '/api/training/start', {
        method: 'POST'
      }, 5000);
    } catch (err) {
      console.error('[SOUL-LLM] notifyTrainer failed (is SOUL-LLM API running?):', err.message);
    }
  }

  // Check connectivity to Ollama, SOUL-LLM, or local OpenAI endpoint
  async checkHealth() {
    let availableModels = [];
    let soulLlmOnline = false;

    // Check SOUL-LLM on port 8000 (always listed; used only as fallback)
    try {
      const soulRes = await this.makeRequest('http://localhost:8000/health', { method: 'GET' }, 3000);
      const soulData = JSON.parse(soulRes);
      if (soulData.status === 'healthy') {
        soulLlmOnline = true;
        availableModels.push('SOUL-LLM (Scratch PyTorch)');
      }
    } catch {}

    // Probe the preferred engine (normal case: Ollama on 11434)
    const isOllama = this.preferred.provider === 'ollama';
    const testPath = isOllama ? '/api/tags' : '/v1/models';
    let preferredUrl;
    try {
      preferredUrl = new URL(this.preferred.endpoint);
    } catch {
      preferredUrl = new URL('http://localhost:11434');
    }

    try {
      const response = await this.makeRequest(preferredUrl.origin + testPath, { method: 'GET' }, 5000);
      const data = JSON.parse(response);

      if (isOllama && data.models) {
        availableModels.push(...data.models.map(m => m.name));
      } else if (data.data) {
        availableModels.push(...data.data.map(m => m.id));
      }

      // Primary engine is back — revert if we had failed over to SOUL-LLM
      if (this.provider !== this.preferred.provider ||
          this.endpoint !== this.preferred.endpoint ||
          this.model !== this.preferred.model) {
        this.provider = this.preferred.provider;
        this.endpoint = this.preferred.endpoint;
        this.model = this.preferred.model;
        console.log(`[Engine] ${this.model} back online — reverted from SOUL-LLM fallback`);
      }

      return {
        online: true,
        endpoint: this.endpoint,
        provider: this.provider,
        activeModel: this.model,
        models: availableModels
      };
    } catch (err) {
      if (soulLlmOnline) {
        this.provider = 'soul-llm';
        this.endpoint = 'http://localhost:8000';
        this.model = 'SOUL-LLM (Scratch PyTorch)';
        return {
          online: true,
          endpoint: this.endpoint,
          provider: this.provider,
          activeModel: this.model,
          models: availableModels
        };
      }
      return {
        online: false,
        endpoint: this.endpoint,
        provider: this.provider,
        activeModel: this.model,
        models: availableModels,
        error: `Could not connect to ${this.endpoint} (${err.message}). Demo fallback available.`
      };
    }
  }

  // Generic HTTP fetch helper with no external dependencies
  makeRequest(targetUrl, options = {}, timeoutMs = 60000) {
    return new Promise((resolve, reject) => {
      const parsed = new URL(targetUrl);
      const client = parsed.protocol === 'https:' ? https : http;

      const req = client.request(parsed, {
        method: options.method || 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {})
        },
        timeout: timeoutMs
      }, (res) => {
        let body = '';
        res.on('data', chunk => body += chunk);
        res.on('end', () => {
          if (res.statusCode >= 200 && res.statusCode < 300) {
            resolve(body);
          } else {
            reject(new Error(`HTTP ${res.statusCode}: ${body || res.statusMessage}`));
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error(`Request timed out after ${timeoutMs}ms`));
      });

      if (options.body) {
        req.write(typeof options.body === 'string' ? options.body : JSON.stringify(options.body));
      }
      req.end();
    });
  }

  // Stream raw HTTP response
  streamPost(targetUrl, payload, onChunk, onEnd, onError) {
    const parsed = new URL(targetUrl);
    const client = parsed.protocol === 'https:' ? https : http;

    const req = client.request(parsed, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    }, (res) => {
      if (res.statusCode < 200 || res.statusCode >= 300) {
        let errBody = '';
        res.on('data', chunk => errBody += chunk);
        res.on('end', () => onError(new Error(`HTTP ${res.statusCode}: ${errBody}`)));
        return;
      }

      let buffer = '';
      res.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep partial line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          onChunk(trimmed);
        }
      });

      res.on('end', () => {
        if (buffer.trim()) {
          onChunk(buffer.trim());
        }
        onEnd();
      });
    });

    req.on('error', onError);
    req.write(JSON.stringify(payload));
    req.end();
  }

  // Extract tool calls from markdown code blocks or structured text
  parseToolCalls(text) {
    const toolCalls = [];

    // Format 1: ```tool_call { ... } ```
    const codeBlockRegex = /```(?:tool_call|tool|json)?\s*(\{[\s\S]*?"name"\s*:\s*"[\s\S]*?\})\s*```/g;
    let match;
    while ((match = codeBlockRegex.exec(text)) !== null) {
      try {
        const parsed = JSON.parse(match[1]);
        if (parsed.name) {
          toolCalls.push({
            name: parsed.name,
            arguments: parsed.arguments || parsed.parameters || {}
          });
        }
      } catch {
        // continue
      }
    }

    // Format 2: <tool_call>{"name": ...}</tool_call>
    const tagRegex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
    while ((match = tagRegex.exec(text)) !== null) {
      try {
        const parsed = JSON.parse(match[1]);
        if (parsed.name) {
          toolCalls.push({
            name: parsed.name,
            arguments: parsed.arguments || parsed.parameters || {}
          });
        }
      } catch {
        // continue
      }
    }

    return toolCalls;
  }

  // Extract thoughts: <thought>...</thought>
  extractThoughts(text) {
    const thoughts = [];
    const thoughtRegex = /<thought>([\s\S]*?)<\/thought>/g;
    let match;
    while ((match = thoughtRegex.exec(text)) !== null) {
      thoughts.push(match[1].trim());
    }
    const cleanContent = text.replace(/<thought>[\s\S]*?<\/thought>/g, '').trim();
    return { thoughts, cleanContent };
  }

  // Fallback intelligent simulation when Ollama is offline
  async runFallbackSimulatedAgent(userPrompt, callbacks) {
    callbacks.onThought && callbacks.onThought('Local Ollama server is not detected on port 11434. Running in intelligent offline diagnostic mode.');
    
    // Check if user is asking to calculate something
    const mathMatch = userPrompt.match(/(\d+\s*[\+\-\*\/]\s*\d+)/);
    if (mathMatch && userPrompt.toLowerCase().includes('calc')) {
      const expr = mathMatch[1];
      callbacks.onToolCall && callbacks.onToolCall({ name: 'calculate', arguments: { expression: expr } });
      const res = await executeTool('calculate', { expression: expr });
      callbacks.onToolResult && callbacks.onToolResult({ name: 'calculate', result: res });
      
      const response = `I evaluated the expression \`${expr}\` using the local offline calculation tool:\n\n**Result:** \`${res.result}\`\n\n*(Note: To enable autonomous local LLM generation, start Ollama with your RTX 4060 GPU by running \`ollama serve\`)*.`;
      callbacks.onToken && callbacks.onToken(response);
      return response;
    }

    // Check if user is asking for system info
    if (userPrompt.toLowerCase().includes('system') || userPrompt.toLowerCase().includes('gpu') || userPrompt.toLowerCase().includes('specs')) {
      callbacks.onToolCall && callbacks.onToolCall({ name: 'getSystemInfo', arguments: {} });
      const res = await executeTool('getSystemInfo', {});
      callbacks.onToolResult && callbacks.onToolResult({ name: 'getSystemInfo', result: res });

      const response = `### 🖥️ Local System Telemetry\n\n` +
        `- **OS**: Windows (${res.platform} ${res.release})\n` +
        `- **CPU**: ${res.cpuModel} (${res.cpuCores} threads)\n` +
        `- **RAM**: ${res.ram.usedGb} / ${res.ram.totalGb} (${res.ram.percentUsed} used)\n` +
        `- **GPU**: ${res.gpu}\n` +
        `- **Workspace**: \`${res.workspacePath}\`\n\n` +
        `> **RTX 4060 Ready**: Your 8GB NVIDIA GPU is detected. Run \`.\\setup-ollama.ps1\` to install Ollama and pull the recommended \`llama3.1:8b\` model!`;
      callbacks.onToken && callbacks.onToken(response);
      return response;
    }

    // Default fallback message
    const msg = `Hello! I am your **100% Offline AI Agent**.\n\n` +
      `I am ready to run fully autonomous reasoning using your **NVIDIA GeForce RTX 4060 GPU**.\n\n` +
      `### To activate Local LLM Power:\n` +
      `1. Open PowerShell and run: \`powershell .\\setup-ollama.ps1\` (or run \`winget install Ollama.Ollama\`)\n` +
      `2. Download a high-performance 8B model: \`ollama pull llama3.1:8b\` or \`ollama pull qwen2.5:7b\`\n` +
      `3. Refresh or click **Check Connection** in the top bar.\n\n` +
      `### What I can do right now:\n` +
      `- You can test tool executions like \`calculate\` ("calculate 256 * 48") or \`getSystemInfo\` ("check system specs")\n` +
      `- Manage workspace files in the **Workspace** tab\n` +
      `- Build your offline knowledge index in the **Local Knowledge (RAG)** tab!`;

    callbacks.onToken && callbacks.onToken(msg);
    return msg;
  }

  // Pre-load the model into VRAM so the first chat request responds fast
  async warmup() {
    const health = await this.checkHealth();
    if (!health.online || health.provider !== 'ollama') return false;
    if (health.models.length === 0) return false;

    try {
      const url = new URL(this.endpoint);
      await this.makeRequest(url.origin + '/api/chat', {
        method: 'POST',
        body: {
          model: this.model,
          messages: [{ role: 'user', content: 'ping' }],
          stream: false,
          keep_alive: '30m',
          options: { num_predict: 1 }
        }
      }, 120000);
      return true;
    } catch {
      return false;
    }
  }

  // Autonomous ReAct loop execution
  async runChat(messages, callbacks = {}) {
    // 1. Check health
    const health = await this.checkHealth();
    if (!health.online) {
      const lastUserMsg = messages.filter(m => m.role === 'user').pop()?.content || '';
      return await this.runFallbackSimulatedAgent(lastUserMsg, callbacks);
    }

    const systemPrompt = buildSystemPrompt(tools, this.customInstructions);
    const conversation = [
      { role: 'system', content: systemPrompt },
      ...messages
    ];

    let iteration = 0;
    let finalAnswer = '';
    let succeeded = false;

    try {
      // wrap loop body in try so capture never breaks the chat
      await this._reactLoop(conversation, callbacks, (state) => {
        iteration = state.iteration;
        finalAnswer = state.finalAnswer;
        succeeded = !!state.succeeded;
      });
    } catch (err) {
      callbacks.onError && callbacks.onError(err);
      return `Error calling local LLM: ${err.message}`;
    }

    // Distill every successful chat into SOUL-LLM training data
    if (succeeded) {
      this.captureConversation(messages, finalAnswer);
    }

    callbacks.onFinish && callbacks.onFinish({
      response: finalAnswer,
      iterations: iteration
    });

    return finalAnswer;
  }

  async _reactLoop(conversation, callbacks, report) {
    let iteration = 0;
    let finalAnswer = '';
    let succeeded = false;

    while (iteration < this.maxIterations) {
      iteration++;

      // Request completion from local model
      let rawModelOutput = '';
      try {
        rawModelOutput = await this.callLocalModel(conversation, callbacks);
      } catch (err) {
        if (err.message && err.message.includes('404') && err.message.includes('not found')) {
          const friendlyMsg = `### ⚠️ Model Not Downloaded Yet\n\n` +
            `The model \`${this.model}\` has not been downloaded to your local machine yet.\n\n` +
            `- **Quick Fix:** Go to the **Models & Engine** tab in the sidebar, or run in PowerShell:\n` +
            `  \`\`\`powershell\n  ollama pull ${this.model}\n  \`\`\`\n` +
            `- **Currently Downloading:** \`qwen2.5:7b\` is in progress.\n` +
            `Once the download completes, refresh the page or select the downloaded model in the top bar!`;
          callbacks.onToken && callbacks.onToken(friendlyMsg);
          report({ iteration, finalAnswer: friendlyMsg, succeeded: false });
          return;
        }
        callbacks.onError && callbacks.onError(err);
        report({ iteration, finalAnswer: `Error calling local LLM: ${err.message}`, succeeded: false });
        return;
      }

      // Check for thoughts
      const { thoughts, cleanContent } = this.extractThoughts(rawModelOutput);
      for (const th of thoughts) {
        callbacks.onThought && callbacks.onThought(th);
      }

      // Check for tool calls
      const toolCalls = this.parseToolCalls(rawModelOutput);

      // If no tool call was made, the agent finished its reasoning and delivered the final answer!
      if (toolCalls.length === 0) {
        finalAnswer = cleanContent || rawModelOutput;
        succeeded = true;
        break;
      }

      // Append assistant's response to conversation
      conversation.push({
        role: 'assistant',
        content: rawModelOutput
      });

      // Execute each detected tool call
      for (const call of toolCalls) {
        callbacks.onToolCall && callbacks.onToolCall(call);

        let result;
        try {
          result = await executeTool(call.name, call.arguments);
        } catch (err) {
          result = { error: err.message };
        }

        callbacks.onToolResult && callbacks.onToolResult({
          name: call.name,
          result
        });

        // Feed tool result back to the model
        conversation.push({
          role: 'user',
          content: `[TOOL_OBSERVATION for ${call.name}]:\n${JSON.stringify(result, null, 2)}\n\nNow continue reasoning. If you have enough information, answer the user; otherwise, call the next tool.`
        });
      }
    }

    report({ iteration, finalAnswer, succeeded });
  }

  // Call Ollama, SOUL-LLM, or OpenAI compatible endpoint
  callLocalModel(conversation, callbacks) {
    return new Promise((resolve, reject) => {
      const isSoulLlm = this.model.toLowerCase().includes('soul-llm') || this.provider === 'soul-llm';

      if (isSoulLlm) {
        // Direct integration with local SOUL-LLM PyTorch backend on port 8000
        const payload = {
          model: 'soul-llm',
          messages: conversation,
          temperature: this.temperature,
          stream: true
        };

        let fullText = '';
        this.streamPost('http://localhost:8000/v1/chat/completions', payload, (line) => {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace(/^data: /, '').trim();
            if (dataStr === '[DONE]') return;
            try {
              const parsed = JSON.parse(dataStr);
              const delta = parsed.choices?.[0]?.delta?.content || '';
              if (delta) {
                fullText += delta;
                callbacks.onToken && callbacks.onToken(delta);
              }
            } catch {}
          }
        }, () => {
          resolve(fullText);
        }, (err) => {
          reject(err);
        });

      } else if (this.provider === 'ollama') {
        const payload = {
          model: this.model,
          messages: conversation,
          stream: true,
          keep_alive: '30m',
          options: {
            temperature: this.temperature
          }
        };

        let fullText = '';
        this.streamPost(`${this.endpoint}/api/chat`, payload, (jsonLine) => {
          try {
            const parsed = JSON.parse(jsonLine);
            if (parsed.message && parsed.message.content) {
              const delta = parsed.message.content;
              fullText += delta;
              callbacks.onToken && callbacks.onToken(delta);
            }
          } catch {
            // non-json line
          }
        }, () => {
          resolve(fullText);
        }, (err) => {
          reject(err);
        });

      } else {
        // OpenAI compatible (/v1/chat/completions)
        const payload = {
          model: this.model,
          messages: conversation,
          temperature: this.temperature,
          stream: true
        };

        let fullText = '';
        this.streamPost(`${this.endpoint}/v1/chat/completions`, payload, (line) => {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace(/^data: /, '').trim();
            if (dataStr === '[DONE]') return;
            try {
              const parsed = JSON.parse(dataStr);
              const delta = parsed.choices?.[0]?.delta?.content || '';
              if (delta) {
                fullText += delta;
                callbacks.onToken && callbacks.onToken(delta);
              }
            } catch {}
          }
        }, () => {
          resolve(fullText);
        }, (err) => {
          reject(err);
        });
      }
    });
  }
}

const defaultAgent = new LocalAgentEngine();

module.exports = {
  LocalAgentEngine,
  defaultAgent
};
