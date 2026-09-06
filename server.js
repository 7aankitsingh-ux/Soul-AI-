const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { defaultAgent } = require('./agent/core');
const { tools, executeTool, WORKSPACE_DIR, resolveWorkspacePath } = require('./agent/tools');
const { ragEngine, KNOWLEDGE_DIR } = require('./agent/rag');
const auth = require('./auth');

const SOUL_LLM_API = process.env.SOUL_LLM_API || 'http://localhost:8000';

// Thin proxy to the SOUL-LLM trainer API (FastAPI on :8000). The trainer
// controls (start/pause/resume/stop/status/config) live there; the web UI
// reaches them through this same-origin path.
async function proxyToSoulLlm(path, options = {}) {
  try {
    const res = await fetch(SOUL_LLM_API + path, {
      method: options.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: options.body ? JSON.stringify(options.body) : undefined
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    if (!res.ok) return { success: false, error: (data && data.error) || `HTTP ${res.status}` };
    return data;
  } catch (err) {
    return { success: false, error: `SOUL-LLM API unreachable: ${err.message}` };
  }
}

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ─── Cookie / auth middleware ───
function currentUser(req) {
  const cookies = auth.parseCookies(req);
  return auth.usernameFromToken(cookies.soulai);
}

function setAuthCookie(res, token) {
  res.setHeader('Set-Cookie', `soulai=${token}; HttpOnly; Path=/; SameSite=Lax; Max-Age=2592000`);
}

function clearAuthCookie(res) {
  res.setHeader('Set-Cookie', 'soulai=; HttpOnly; Path=/; SameSite=Lax; Max-Age=0');
}

// ─── 0. Auth endpoints ───
app.post('/api/auth/register', (req, res) => {
  const username = auth.sanitizeUsername(req.body.username);
  const password = typeof req.body.password === 'string' ? req.body.password : '';
  if (!username) return res.status(400).json({ success: false, error: 'Username: 3-20 chars, letters/numbers/underscore' });
  if (password.length < 4) return res.status(400).json({ success: false, error: 'Password must be at least 4 characters' });

  const result = auth.registerUser(username, password);
  if (result.error) return res.status(409).json({ success: false, error: result.error });

  const token = auth.createSession(username);
  setAuthCookie(res, token);
  res.json({ success: true, user: result.user });
});

app.post('/api/auth/login', (req, res) => {
  const username = String(req.body.username || '');
  const password = String(req.body.password || '');
  const result = auth.loginUser(username, password);
  if (result.error) return res.status(401).json({ success: false, error: result.error });

  const token = auth.createSession(username);
  setAuthCookie(res, token);
  res.json({ success: true, user: result.user });
});

app.post('/api/auth/logout', (req, res) => {
  const cookies = auth.parseCookies(req);
  if (cookies.soulai) auth.destroySession(cookies.soulai);
  clearAuthCookie(res);
  res.json({ success: true });
});

app.get('/api/auth/me', (req, res) => {
  const username = currentUser(req);
  if (!username) return res.json({ success: true, user: null });
  res.json({ success: true, user: { username } });
});

// ─── 0b. Chat history ───
app.get('/api/chats', (req, res) => {
  const username = currentUser(req);
  if (!username) return res.status(401).json({ success: false, error: 'Not logged in' });
  res.json({ success: true, chats: auth.getChats(username) });
});

app.post('/api/chats', (req, res) => {
  const username = currentUser(req);
  if (!username) return res.status(401).json({ success: false, error: 'Not logged in' });
  const chat = auth.saveChat(username, req.body.chat);
  if (chat.error) return res.status(400).json({ success: false, error: chat.error });
  res.json({ success: true, chat });
});

app.delete('/api/chats/:id', (req, res) => {
  const username = currentUser(req);
  if (!username) return res.status(401).json({ success: false, error: 'Not logged in' });
  auth.deleteChat(username, req.params.id);
  res.json({ success: true });
});

// 1. System & Engine Status
app.get('/api/status', async (req, res) => {
  try {
    const health = await defaultAgent.checkHealth();
    const sysInfo = await executeTool('getSystemInfo', {});
    res.json({
      success: true,
      agent: health,
      system: sysInfo
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 2. Settings update
app.post('/api/settings', (req, res) => {
  try {
    const { provider, endpoint, model, temperature, customInstructions, maxIterations, retrainAfter, collectMode } = req.body;
    defaultAgent.setOptions({
      provider,
      endpoint,
      model,
      temperature,
      customInstructions,
      maxIterations,
      retrainAfter,
      collectMode
    });
    res.json({ success: true, message: 'Settings updated successfully' });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 2b. SOUL-LLM distillation trainer (proxied to the SOUL-LLM API on :8000)
app.get('/api/trainer/status', async (req, res) => {
  const data = await proxyToSoulLlm('/api/training/status');
  data.model = defaultAgent.model;
  data.collectMode = defaultAgent.collectMode;
  data.trainAfter = defaultAgent.retrainAfter;
  if (!data.dataset_size && data.dataset_size !== 0) data.dataset_size = defaultAgent.datasetSize();
  res.json(data);
});

app.post('/api/trainer/start', async (req, res) => res.json(await proxyToSoulLlm('/api/training/start', { method: 'POST', body: {} })));
app.post('/api/trainer/pause', async (req, res) => res.json(await proxyToSoulLlm('/api/training/pause', { method: 'POST', body: {} })));
app.post('/api/trainer/resume', async (req, res) => res.json(await proxyToSoulLlm('/api/training/resume', { method: 'POST', body: {} })));
app.post('/api/trainer/stop', async (req, res) => res.json(await proxyToSoulLlm('/api/training/stop', { method: 'POST', body: {} })));
app.post('/api/trainer/evaluate', async (req, res) => res.json(await proxyToSoulLlm('/api/training/evaluate', { method: 'POST', body: {} })));
app.post('/api/trainer/model', async (req, res) => res.json(await proxyToSoulLlm('/api/model/select', { method: 'POST', body: req.body || {} })));

app.get('/api/trainer/config', async (req, res) => res.json(await proxyToSoulLlm('/api/training/config')));
app.post('/api/trainer/config', async (req, res) => res.json(await proxyToSoulLlm('/api/training/config', { method: 'POST', body: req.body || {} })));

// Manual collection: save one Q&A pair into the teacher dataset (any mode)
app.post('/api/trainer/capture', async (req, res) => {
  const { prompt, response } = req.body || {};
  if (!prompt || !response) return res.status(400).json({ success: false, error: 'prompt and response required' });
  const ok = await defaultAgent.saveManualExample(prompt, response);
  res.json({ success: ok, datasetSize: defaultAgent.datasetSize() });
});

// 2b. Model Pull Endpoint (runs ollama pull <model>)
app.post('/api/models/pull', (req, res) => {
  const { model } = req.body;
  if (!model) return res.status(400).json({ error: 'Model name required (e.g. qwen2.5:7b, llama3.1:8b)' });

  const { spawn } = require('child_process');
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const child = spawn('ollama', ['pull', model], { shell: true });

  child.stdout.on('data', (chunk) => {
    res.write(`data: ${JSON.stringify({ text: chunk.toString() })}\n\n`);
  });

  child.stderr.on('data', (chunk) => {
    res.write(`data: ${JSON.stringify({ text: chunk.toString() })}\n\n`);
  });

  child.on('close', (code) => {
    res.write(`data: ${JSON.stringify({ done: true, code })}\n\n`);
    res.end();
  });

  child.on('error', (err) => {
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  });
});

// 3. SSE Chat Endpoint with Real-Time ReAct Loop
app.post('/api/chat', async (req, res) => {
  const { messages } = req.body;
  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'messages array required' });
  }

  // Setup Server-Sent Events (SSE)
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders && res.flushHeaders();

  const sendEvent = (type, data) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    await defaultAgent.runChat(messages, {
      onThought: (thought) => {
        sendEvent('thought', { thought });
      },
      onToken: (token) => {
        sendEvent('token', { token });
      },
      onToolCall: (call) => {
        sendEvent('tool_call', call);
      },
      onToolResult: (result) => {
        sendEvent('tool_result', result);
      },
      onFinish: (summary) => {
        sendEvent('finish', summary);
      },
      onError: (err) => {
        sendEvent('error', { message: err.message });
      }
    });

    sendEvent('done', { complete: true });
    res.end();
  } catch (err) {
    sendEvent('error', { message: err.message });
    res.end();
  }
});

// 4. Tools Registry & Direct Execution
app.get('/api/tools', (req, res) => {
  const list = tools.map(t => ({
    name: t.name,
    description: t.description,
    parameters: t.parameters
  }));
  res.json({ success: true, tools: list });
});

app.post('/api/tools/execute', async (req, res) => {
  const { name, args } = req.body;
  if (!name) {
    return res.status(400).json({ error: 'Tool name required' });
  }
  const result = await executeTool(name, args || {});
  res.json({ success: true, tool: name, result });
});

// 5. Workspace File Management
app.get('/api/workspace', (req, res) => {
  try {
    const list = fs.readdirSync(WORKSPACE_DIR, { withFileTypes: true }).map(e => ({
      name: e.name,
      isDirectory: e.isDirectory(),
      size: e.isFile() ? fs.statSync(path.join(WORKSPACE_DIR, e.name)).size : null,
      updatedAt: fs.statSync(path.join(WORKSPACE_DIR, e.name)).mtime
    }));
    res.json({ success: true, workspace: WORKSPACE_DIR, files: list });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.get('/api/workspace/file', (req, res) => {
  const filename = req.query.name;
  if (!filename) return res.status(400).json({ error: 'File name required' });
  const fullPath = resolveWorkspacePath(filename);
  if (!fs.existsSync(fullPath)) return res.status(404).json({ error: 'File not found' });
  try {
    const content = fs.readFileSync(fullPath, 'utf8');
    res.json({ success: true, name: filename, content });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post('/api/workspace/file', (req, res) => {
  const { name, content } = req.body;
  if (!name) return res.status(400).json({ error: 'File name required' });
  const fullPath = resolveWorkspacePath(name);
  try {
    fs.writeFileSync(fullPath, content || '', 'utf8');
    res.json({ success: true, message: `Saved ${name}` });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.delete('/api/workspace/file', (req, res) => {
  const filename = req.query.name;
  if (!filename) return res.status(400).json({ error: 'File name required' });
  const fullPath = resolveWorkspacePath(filename);
  if (!fs.existsSync(fullPath)) return res.status(404).json({ error: 'File not found' });
  try {
    fs.unlinkSync(fullPath);
    res.json({ success: true, message: `Deleted ${filename}` });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 6. RAG Document Management
app.get('/api/rag', (req, res) => {
  res.json({ success: true, ...ragEngine.getStats() });
});

app.post('/api/rag/upload', (req, res) => {
  const { filename, content } = req.body;
  if (!filename || content === undefined) {
    return res.status(400).json({ error: 'filename and content required' });
  }
  try {
    const fullPath = path.join(KNOWLEDGE_DIR, filename);
    fs.writeFileSync(fullPath, content, 'utf8');
    ragEngine.addDocument(filename, content);
    res.json({ success: true, message: `Indexed ${filename} in knowledge base` });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.delete('/api/rag/file', (req, res) => {
  const filename = req.query.name;
  if (!filename) return res.status(400).json({ error: 'File name required' });
  const result = ragEngine.removeDocument(filename);
  res.json(result);
});

// Start Server
auth.ensureStorage();
app.listen(PORT, () => {
  console.log(`\n=================================================`);
  console.log(`  🚀 OFFLINE AI AGENT SERVER RUNNING`);
  console.log(`  URL: http://localhost:${PORT}`);
  console.log(`  Workspace: ${WORKSPACE_DIR}`);
  console.log(`=================================================\n`);

  // Pre-load the model so the first chat request doesn't time out on VRAM load
  defaultAgent.warmup().then(ok => {
    if (ok) {
      console.log(`  ✅ Model "${defaultAgent.model}" pre-loaded & kept warm (keep_alive 30m)`);
    } else {
      console.log(`  ℹ️  Model warmup skipped (Ollama offline or model unavailable)`);
    }
  }).catch(() => {});
});
