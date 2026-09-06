// Soul AI — Clean & Simple Client Application

let conversationHistory = [];
let availableTools = [];
let activeTab = 'chat';
let currentUser = null;
let currentChatId = null;
let currentChatTitle = null;
let lastExchange = null;

// Simple lightweight offline Markdown parser
function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Code blocks: ```lang\ncode\n```
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline code: `code`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers: ### Header
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Bold & Italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Blockquotes: > quote
  html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // Lists: - item
  html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Paragraph line breaks
  const parts = html.split(/(<pre>[\s\S]*?<\/pre>)/);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  }

  return parts.join('');
}

// DOM Elements
const chatHistory = document.getElementById('chat-history');
const welcomeScreen = document.getElementById('welcome-screen');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const footerModel = document.getElementById('footer-model');
const headerModelSelect = document.getElementById('header-model-select');
const refreshStatusBtn = document.getElementById('refresh-status-btn');
const chatSearchInput = document.getElementById('chat-search-input');

// Use the user's custom icon if they dropped it into /public (icon.png|jpg|jpeg),
// otherwise fall back to the built-in monochrome favicon.svg.
async function initIcon() {
  const candidates = ['icon.png', 'icon.jpg', 'icon.jpeg'];
  const brandIcon = document.getElementById('brand-icon');
  const authIcon = document.getElementById('auth-icon');
  for (const c of candidates) {
    try {
      const res = await fetch(c, { method: 'HEAD' });
      if (res.ok) {
        if (brandIcon) brandIcon.src = c;
        if (authIcon) authIcon.src = c;
        let link = document.querySelector('link[rel="icon"]');
        if (!link) {
          link = document.createElement('link');
          link.rel = 'icon';
          document.head.appendChild(link);
        }
        link.href = c;
        return;
      }
    } catch (e) { /* keep trying */ }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initIcon();
  setupNavigation();
  setupChat();
  setupWorkspace();
  setupRag();
  setupTools();
  setupSettings();
  setupTrainer();
  setupAuth();
  checkStatus();
  initAuth();

  if (chatSearchInput) {
    chatSearchInput.addEventListener('input', applyChatSearch);
  }

  // Auto-resize input
  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 160) + 'px';
  });
});

// Tab Navigation
function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetTab = item.dataset.tab;
      navItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');

      document.querySelectorAll('.tab-view').forEach(tab => tab.classList.remove('active'));
      document.getElementById(`tab-${targetTab}`).classList.add('active');
      activeTab = targetTab;

      if (targetTab === 'workspace') loadWorkspaceFiles();
      if (targetTab === 'rag') loadRagStats();
      if (targetTab === 'tools') loadTools();
    });
  });
}

// Check Server & GPU Status
async function checkStatus() {
  statusText.textContent = 'Connecting...';
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    if (data.agent && data.agent.online) {
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Soul AI Online';

      if (data.agent.models && data.agent.models.length > 0) {
        const currentVal = headerModelSelect.value;
        headerModelSelect.innerHTML = '';
        data.agent.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          if (m === data.agent.activeModel || m === currentVal) opt.selected = true;
          headerModelSelect.appendChild(opt);
        });
      }

      const isSoulLlm = (data.agent.activeModel || headerModelSelect.value || '').includes('SOUL-LLM');
      footerModel.textContent = isSoulLlm
        ? 'SOUL-LLM • Scratch PyTorch'
        : `${data.agent.activeModel || headerModelSelect.value} • RTX 4060`;
    } else {
      statusDot.className = 'status-dot';
      statusText.textContent = 'Standby (Offline)';
      footerModel.textContent = 'Fallback Mode';
    }
  } catch (err) {
    statusDot.className = 'status-dot';
    statusText.textContent = 'Offline';
  }
}

if (refreshStatusBtn) refreshStatusBtn.addEventListener('click', checkStatus);

// Chat Setup
function setupChat() {
  sendBtn.addEventListener('click', handleSendMessage);
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  newChatBtn.addEventListener('click', () => newChat());

  headerModelSelect.addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    const isSoul = selectedModel.includes('SOUL-LLM') || selectedModel === 'soul-llm';
    footerModel.textContent = isSoul ? 'SOUL-LLM • Scratch PyTorch' : `${selectedModel} • RTX 4060`;
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        provider: isSoul ? 'soul-llm' : 'ollama'
      })
    });
  });
}

function resetChat() {
  conversationHistory = [];
  currentChatId = null;
  currentChatTitle = null;
  chatHistory.innerHTML = '';
  if (welcomeScreen) {
    chatHistory.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'flex';
  }
  userInput.value = '';
  userInput.style.height = 'auto';
  userInput.focus();
}

// Start a fresh chat, saving the current one first
async function newChat() {
  await saveCurrentChat();
  resetChat();
  syncChatActiveState();
}

function syncChatActiveState() {
  document.querySelectorAll('.chat-item').forEach(item => {
    item.classList.toggle('active', item.dataset.id === currentChatId);
  });
}

window.sendQuickPrompt = (text) => {
  userInput.value = text;
  handleSendMessage();
};

async function handleSendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Hide welcome screen on first message
  if (welcomeScreen && welcomeScreen.parentNode === chatHistory) {
    welcomeScreen.style.display = 'none';
  }

  userInput.value = '';
  userInput.style.height = 'auto';
  sendBtn.disabled = true;

  // Append user message
  appendUserMessage(text);
  conversationHistory.push({ role: 'user', content: text });

  // Create assistant message container
  const assistantMsg = createAssistantMessage();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: conversationHistory })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = '';
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      let currentEvent = null;

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.replace('event: ', '').trim();
        } else if (line.startsWith('data: ')) {
          const rawData = line.replace('data: ', '').trim();
          try {
            const data = JSON.parse(rawData);

            if (currentEvent === 'thought') {
              assistantMsg.addThought(data.thought);
            } else if (currentEvent === 'tool_call') {
              assistantMsg.addToolCall(data.name, data.arguments);
            } else if (currentEvent === 'tool_result') {
              assistantMsg.updateToolResult(data.name, data.result);
            } else if (currentEvent === 'token') {
              accumulatedText += data.token;
              assistantMsg.updateContent(accumulatedText);
            } else if (currentEvent === 'finish') {
              if (data.response && !accumulatedText) {
                accumulatedText = data.response;
                assistantMsg.updateContent(accumulatedText);
              }
            } else if (currentEvent === 'error') {
              assistantMsg.updateContent(`\n\n**Error:** ${data.message}`);
            }
          } catch {}
        }
      }
    }

    if (accumulatedText) {
      conversationHistory.push({ role: 'assistant', content: accumulatedText });
      lastExchange = { prompt: text, response: accumulatedText };
    }
    await saveCurrentChat();

  } catch (err) {
    assistantMsg.updateContent(`**Error:** ${err.message}. Please check if the local server is running.`);
    await saveCurrentChat();
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}

// Save the in-progress conversation if the user closes the tab
window.addEventListener('beforeunload', () => {
  const chat = localChatSnapshot();
  if (chat) saveChatLocal(chat);
});

// ─── Chat history persistence ───
// Chats always save locally (browser storage) so they survive even as a guest.
// Logged-in users also get them saved on the server via /api/chats.

const HISTORY_KEY = 'soulai_chats_v1';

function getLocalChats() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const data = raw ? JSON.parse(raw) : { chats: [] };
    return Array.isArray(data.chats) ? data.chats : [];
  } catch { return []; }
}

function setLocalChats(chats) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify({ chats })); } catch {}
}

function saveChatLocal(chat) {
  const chats = getLocalChats();
  const idx = chats.findIndex(c => c.id === chat.id);
  if (idx >= 0) chats[idx] = chat; else chats.push(chat);
  setLocalChats(chats);
}

function localChatSnapshot() {
  if (!conversationHistory.length) return null;
  const firstUser = conversationHistory.find(m => m.role === 'user');
  if (!firstUser) return null;
  const id = currentChatId || `chat_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  return {
    id,
    title: currentChatTitle || firstUser.content.slice(0, 42),
    messages: conversationHistory,
    updatedAt: new Date().toISOString()
  };
}

// Persist the current conversation locally, and on the server when logged in
async function saveCurrentChat() {
  const chat = localChatSnapshot();
  if (!chat) return;
  currentChatId = chat.id;
  currentChatTitle = chat.title;

  saveChatLocal(chat);

  if (currentUser) {
    try {
      await fetch('/api/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat })
      });
    } catch {}
  }
  loadChats();
}

// After login, carry any locally saved (guest) chats into the account
async function migrateLocalChatsToServer() {
  const local = getLocalChats();
  if (!local.length) return;
  try {
    const res = await fetch('/api/chats');
    const data = await res.json();
    if (!data.success) return;
    const serverIds = new Set((data.chats || []).map(c => c.id));
    for (const chat of local) {
      if (serverIds.has(chat.id)) continue;
      await fetch('/api/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat })
      });
    }
    setLocalChats([]);
  } catch {}
}

// Searchable chat history
let cachedChats = [];

function matchChat(chat, query) {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if ((chat.title || '').toLowerCase().includes(q)) return true;
  if (Array.isArray(chat.messages)) {
    return chat.messages.some(m =>
      typeof m.content === 'string' && m.content.toLowerCase().includes(q)
    );
  }
  return false;
}

function applyChatSearch() {
  const q = chatSearchInput.value;
  const filtered = cachedChats.filter(chat => matchChat(chat, q));
  renderChats(filtered, q);
}

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="msg-avatar user">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
    </div>
    <div class="msg-main">
      <div class="msg-bubble">${renderMarkdown(text)}</div>
    </div>
  `;
  chatHistory.appendChild(row);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function createAssistantMessage() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';

  const mainDiv = document.createElement('div');
  mainDiv.className = 'msg-main';

  const thoughtsDiv = document.createElement('div');
  const toolsDiv = document.createElement('div');
const bubbleDiv = document.createElement('div');
  bubbleDiv.className = 'msg-bubble';
  bubbleDiv.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';

  mainDiv.appendChild(thoughtsDiv);
  mainDiv.appendChild(toolsDiv);
  mainDiv.appendChild(bubbleDiv);

  row.innerHTML = '<div class="msg-avatar ai"><svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 2l2.2 5.2 5.6.3-4.3 3.5 1.4 5.4-4.9-2.8-4.9 2.8 1.4-5.4L4.2 7.5l5.6-.3z"/></svg></div>';
  row.appendChild(mainDiv);

  chatHistory.appendChild(row);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  const toolCards = {};

  return {
    addThought: (thought) => {
      const accordion = document.createElement('details');
      accordion.className = 'thought-accordion';
      accordion.open = false;
      accordion.innerHTML = `
        <summary class="thought-summary">Thought process (click to expand)</summary>
        <div class="thought-body">${thought}</div>
      `;
      thoughtsDiv.appendChild(accordion);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    },
    addToolCall: (name, args) => {
      const pill = document.createElement('div');
      pill.className = 'tool-pill';
      pill.innerHTML = `
        <div class="tool-pill-header">
          <span>Running tool: <strong>${name}</strong></span>
        </div>
        <div class="tool-pill-body">arguments: ${JSON.stringify(args, null, 2)}</div>
      `;
      toolsDiv.appendChild(pill);
      toolCards[name] = pill;
      chatHistory.scrollTop = chatHistory.scrollHeight;
    },
    updateToolResult: (name, result) => {
      const card = toolCards[name];
      if (card) {
        const bodyEl = card.querySelector('.tool-pill-body');
        if (bodyEl) {
          bodyEl.innerHTML = `result: ${JSON.stringify(result, null, 2)}`;
        }
      }
      chatHistory.scrollTop = chatHistory.scrollHeight;
    },
    updateContent: (text) => {
      bubbleDiv.innerHTML = renderMarkdown(text);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  };
}

// Workspace Tab
async function loadWorkspaceFiles() {
  const tbody = document.getElementById('workspace-file-list');
  try {
    const res = await fetch('/api/workspace');
    const data = await res.json();
    const files = data.files || [];

    if (files.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted);">Workspace is currently empty.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    files.forEach(file => {
      const tr = document.createElement('tr');
      const sizeStr = file.isDirectory ? '--' : `${file.size} bytes`;
      tr.innerHTML = `
        <td style="font-family:var(--font-mono);">${file.name}</td>
        <td>${file.isDirectory ? '📁 Folder' : '📄 File'}</td>
        <td>${sizeStr}</td>
        <td>
          <button class="btn-clean" style="padding:4px 8px;" onclick="viewFile('${file.name}')">View</button>
          <button class="btn-clean" style="padding:4px 8px; color:var(--accent-rose);" onclick="deleteWorkspaceFile('${file.name}')">Delete</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="4">Error: ${err.message}</td></tr>`;
  }
}

function setupWorkspace() {
  document.getElementById('refresh-workspace-btn').addEventListener('click', loadWorkspaceFiles);
  document.getElementById('close-file-btn').addEventListener('click', () => {
    document.getElementById('file-viewer-card').style.display = 'none';
  });

  window.viewFile = async (filename) => {
    try {
      const res = await fetch(`/api/workspace/file?name=${encodeURIComponent(filename)}`);
      const data = await res.json();
      document.getElementById('viewing-file-name').textContent = filename;
      document.getElementById('file-viewer-content').textContent = data.content || '(Empty file)';
      document.getElementById('file-viewer-card').style.display = 'block';
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  window.deleteWorkspaceFile = async (filename) => {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
      await fetch(`/api/workspace/file?name=${encodeURIComponent(filename)}`, { method: 'DELETE' });
      loadWorkspaceFiles();
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };
}

// RAG Knowledge Base Tab
async function loadRagStats() {
  const tbody = document.getElementById('rag-table-body');
  const countEl = document.getElementById('rag-doc-count');
  try {
    const res = await fetch('/api/rag');
    const data = await res.json();

    countEl.textContent = data.totalDocuments || 0;
    const docs = data.documents || [];

    if (docs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted);">No documents indexed yet.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    docs.forEach(doc => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-family:var(--font-mono);">${doc.filename}</td>
        <td>${doc.charCount} chars</td>
        <td>${new Date(doc.indexedAt).toLocaleDateString()}</td>
        <td>
          <button class="btn-clean" style="padding:4px 8px; color:var(--accent-rose);" onclick="deleteRagDoc('${doc.filename}')">Remove</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="4">Error: ${err.message}</td></tr>`;
  }
}

function setupRag() {
  const indexBtn = document.getElementById('rag-index-btn');
  const nameInput = document.getElementById('rag-file-name');
  const contentInput = document.getElementById('rag-file-content');

  indexBtn.addEventListener('click', async () => {
    const filename = nameInput.value.trim();
    const content = contentInput.value.trim();
    if (!filename || !content) {
      alert('Please provide a document title and content.');
      return;
    }

    indexBtn.disabled = true;
    indexBtn.textContent = 'Indexing...';

    try {
      await fetch('/api/rag/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, content })
      });
      nameInput.value = '';
      contentInput.value = '';
      loadRagStats();
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      indexBtn.disabled = false;
      indexBtn.textContent = 'Index Document';
    }
  });

  window.deleteRagDoc = async (filename) => {
    if (!confirm(`Delete ${filename}?`)) return;
    try {
      await fetch(`/api/rag/file?name=${encodeURIComponent(filename)}`, { method: 'DELETE' });
      loadRagStats();
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };
}

// Tools Tab
async function loadTools() {
  const container = document.getElementById('tools-container');
  try {
    const res = await fetch('/api/tools');
    const data = await res.json();
    availableTools = data.tools || [];

    container.innerHTML = '';
    availableTools.forEach(tool => {
      const card = document.createElement('div');
      card.className = 'simple-card';
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h3>⚡ ${tool.name}</h3>
          <span style="font-size:11.5px; color:var(--text-muted);">Offline tool</span>
        </div>
        <p style="margin-top:4px;">${tool.description}</p>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<p>Error loading tools: ${err.message}</p>`;
  }
}

function setupTools() {}

// Settings & Pull Model
function setupSettings() {
  const saveBtn = document.getElementById('save-settings-btn');
  const instructions = document.getElementById('cfg-instructions');

  saveBtn.addEventListener('click', async () => {
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';
    try {
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          customInstructions: instructions.value.trim()
        })
      });
      alert('Settings saved!');
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save Instructions';
    }
  });

  window.pullModelDirect = async (modelName) => {
    const box = document.getElementById('pull-status-box');
    box.style.display = 'block';
    box.textContent = `Pulling ${modelName} into offline storage...\n`;

    try {
      const res = await fetch('/api/models/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName })
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.replace('data: ', ''));
              if (data.text) {
                box.textContent += data.text;
                box.scrollTop = box.scrollHeight;
              }
              if (data.done) {
                box.textContent += `\n✔ Successfully pulled ${modelName}!\n`;
                checkStatus();
              }
            } catch {}
          }
        }
      }
    } catch (err) {
      box.textContent += `\n✖ Error: ${err.message}\n`;
    }
  };
}

// ─── Auth & Chat History ───
const authOverlay = document.getElementById('auth-overlay');
const authTitle = document.getElementById('auth-title');
const authSub = document.getElementById('auth-sub');
const authForm = document.getElementById('auth-form');
const authUsername = document.getElementById('auth-username');
const authPassword = document.getElementById('auth-password');
const authError = document.getElementById('auth-error');
const authSubmit = document.getElementById('auth-submit');
const tabLoginBtn = document.getElementById('tab-login-btn');
const tabRegisterBtn = document.getElementById('tab-register-btn');

let authMode = 'login';

function setupAuth() {
  tabLoginBtn.addEventListener('click', () => setAuthMode('login'));
  tabRegisterBtn.addEventListener('click', () => setAuthMode('register'));
  authForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await doAuth();
  });
  const openAuth = (mode) => {
    setAuthMode(mode);
    authOverlay.style.display = 'flex';
  };
  const signInBtn = document.getElementById('signin-btn');
  const signUpBtn = document.getElementById('signup-btn');
  if (signInBtn) signInBtn.addEventListener('click', () => openAuth('login'));
  if (signUpBtn) signUpBtn.addEventListener('click', () => openAuth('register'));
  document.getElementById('logout-btn').addEventListener('click', logout);
}

function setAuthMode(mode) {
  authMode = mode;
  authError.textContent = '';
  tabLoginBtn.classList.toggle('active', mode === 'login');
  tabRegisterBtn.classList.toggle('active', mode === 'register');
  authTitle.textContent = mode === 'login' ? 'Login to Soul AI' : 'Create Account';
  authSub.textContent = mode === 'login'
    ? 'Your chats stay private on this computer.'
    : 'Register to save your chats on this computer.';
  authSubmit.textContent = mode === 'login' ? 'Login' : 'Register';
}

async function doAuth() {
  const username = authUsername.value.trim();
  const password = authPassword.value;
  if (!username || !password) {
    authError.textContent = 'Please fill in both fields.';
    return;
  }
  authSubmit.disabled = true;
  authSubmit.textContent = 'Please wait...';
  try {
    const res = await fetch(`/api/auth/${authMode}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const result = await res.json();
    if (!res.ok || !result.success) {
      authError.textContent = result.error || 'Something went wrong.';
      return;
    }
currentUser = result.user;
    authOverlay.style.display = 'none';
    authPassword.value = '';
    showUserRow();
    resetChat();
    await migrateLocalChatsToServer();
    loadChats();
  } catch {
    authError.textContent = 'Could not reach the local server.';
  } finally {
    authSubmit.disabled = false;
    authSubmit.textContent = authMode === 'login' ? 'Login' : 'Register';
  }
}

function showUserRow() {
  document.getElementById('user-row').style.display = 'flex';
  document.getElementById('guest-row').style.display = 'none';
  document.getElementById('user-name').textContent = currentUser ? currentUser.username : '';
}

function showGuestRow() {
  document.getElementById('user-row').style.display = 'none';
  document.getElementById('guest-row').style.display = 'flex';
}

async function logout() {
  try { await fetch('/api/auth/logout', { method: 'POST' }); } catch {}
  currentUser = null;
  currentChatId = null;
  currentChatTitle = null;
  showGuestRow();
  resetChat();
  loadChats();
}

async function initAuth() {
  try {
    const res = await fetch('/api/auth/me');
    const result = await res.json();
    if (result.success && result.user) {
      currentUser = result.user;
      showUserRow();
      loadChats();
      return;
    }
} catch {}
  setAuthMode('login');
  authOverlay.style.display = 'flex';
  loadChats();
}

async function loadChats() {
  let serverChats = [];
  if (currentUser) {
    try {
      const res = await fetch('/api/chats');
      const result = await res.json();
      if (result.success) serverChats = result.chats || [];
    } catch {}
  }
  const localChats = getLocalChats();
  const serverIds = new Set(serverChats.map(c => c.id));
  cachedChats = serverChats.concat(localChats.filter(c => !serverIds.has(c.id)));
  applyChatSearch();
}

function renderChats(chats, query = '') {
  const list = document.getElementById('chats-list');
  const emptyEl = document.getElementById('chats-empty');
  list.innerHTML = '';
  if (!cachedChats.length) {
    emptyEl.textContent = 'No saved chats yet.';
  } else if (!chats.length) {
    emptyEl.textContent = `No matches for "${query}".`;
  }
  if (!chats.length) {
    list.appendChild(emptyEl);
    emptyEl.style.display = 'block';
    return;
  }
  emptyEl.style.display = 'none';
  chats.slice().sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt)).forEach(chat => {
    const btn = document.createElement('button');
    btn.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
    btn.dataset.id = chat.id;

    const title = document.createElement('span');
    title.className = 'chat-item-title';
    title.textContent = chat.title || 'Untitled';
    btn.appendChild(title);

    const del = document.createElement('button');
    del.className = 'chat-item-del';
    del.textContent = '✕';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      try { await fetch(`/api/chats/${chat.id}`, { method: 'DELETE' }); } catch {}
      if (chat.id === currentChatId) resetChat();
      loadChats();
    });
    btn.appendChild(del);

    btn.addEventListener('click', () => openChat(chat));
    list.appendChild(btn);
  });
}

async function openChat(chat) {
  conversationHistory = Array.isArray(chat.messages) ? chat.messages.slice() : [];
  currentChatId = chat.id;
  currentChatTitle = chat.title;

  chatHistory.innerHTML = '';
  if (welcomeScreen && welcomeScreen.parentNode) welcomeScreen.remove();

  for (const msg of conversationHistory) {
    if (msg.role === 'user') {
      appendUserMessage(msg.content);
    } else if (msg.role === 'assistant') {
      const el = createAssistantMessage();
      el.updateContent(msg.content);
    }
  }
  chatHistory.scrollTop = chatHistory.scrollHeight;
  syncChatActiveState();
}

// --- SOUL-LLM Distillation Trainer panel ---
let trainerTimer = null;

function setupTrainer() {
  document.getElementById('t-start').addEventListener('click', () => trainerDo('start'));
  document.getElementById('t-pause').addEventListener('click', () => trainerDo('pause'));
  document.getElementById('t-resume').addEventListener('click', () => trainerDo('resume'));
  document.getElementById('t-stop').addEventListener('click', () => trainerDo('stop'));
  document.getElementById('t-evaluate').addEventListener('click', trainerEvaluate);
  document.getElementById('t-refresh').addEventListener('click', refreshTrainer);
  document.getElementById('t-save-last').addEventListener('click', trainerSaveLast);

  document.getElementById('t-collect-mode').addEventListener('change', async (e) => {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collectMode: e.target.value })
    });
    await trainerApplyConfig();
  });

  document.getElementById('t-train-after').addEventListener('change', trainerApplyConfig);

  // Auto-refresh status while settings tab is open
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      if (item.dataset.tab === 'settings') {
        refreshTrainer();
        if (trainerTimer) clearInterval(trainerTimer);
        trainerTimer = setInterval(refreshTrainer, 5000);
      } else if (trainerTimer) {
        clearInterval(trainerTimer);
        trainerTimer = null;
      }
    });
  });
  refreshTrainer();
}

function setTrainerLog(msg) {
  const box = document.getElementById('t-log');
  box.style.display = 'block';
  box.textContent = msg;
}

async function trainerDo(action) {
  setTrainerLog(`Requesting "${action}"...`);
  try {
    const res = await fetch(`/api/trainer/${action}`, { method: 'POST' });
    const data = await res.json();
    setTrainerLog(data.error ? `Error: ${data.error}` : data.message || `${action} ok`);
  } catch (err) {
    setTrainerLog(`Error: ${err.message}`);
  }
  refreshTrainer();
}

async function trainerApplyConfig() {
  const trainAfter = parseInt(document.getElementById('t-train-after').value, 10) || 5;
  try {
    await fetch('/api/trainer/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ train_after: trainAfter })
    });
  } catch {}
}

async function trainerEvaluate() {
  setTrainerLog('Running real evaluation (val-loss + generated text)...');
  try {
    const res = await fetch('/api/trainer/evaluate', { method: 'POST' });
    const data = await res.json();
    if (!data.success) { setTrainerLog('Error: ' + (data.error || '')); return; }
    renderTrainerEval(data.result);
    setTrainerLog('Evaluation complete.');
  } catch (err) {
    setTrainerLog('Error: ' + err.message);
  }
  refreshTrainer();
}

function renderTrainerEval(result) {
  const box = document.getElementById('t-eval-box');
  const content = document.getElementById('t-eval-content');
  box.style.display = 'block';
  let html = `<div class="t-muted" style="margin-bottom:8px;">device: ${result.device || '?'} � val loss: ${result.val_loss === null || result.val_loss === undefined ? 'n/a' : result.val_loss}</div>`;
  (result.samples || []).forEach(s => {
    html += `<div class="t-eval-sample"><div class="p">PROMPT: ${s.prompt}</div><div class="r">${s.response || '(empty)'}</div></div>`;
  });
  content.innerHTML = html;
}

async function trainerUseCheckpoint(name) {
  setTrainerLog(`Selecting checkpoint ${name}...`);
  try {
    const res = await fetch('/api/trainer/model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint: name })
    });
    const data = await res.json();
    setTrainerLog(data.error ? `Error: ${data.error}` : `Active checkpoint: ${data.checkpoint}`);
  } catch (err) {
    setTrainerLog(`Error: ${err.message}`);
  }
  refreshTrainer();
}

async function trainerSaveLast() {
  if (!lastExchange) {
    setTrainerLog('No last exchange to save. Ask a question to Qwen first.');
    return;
  }
  try {
    const res = await fetch('/api/trainer/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: lastExchange.prompt, response: lastExchange.response })
    });
    const data = await res.json();
    setTrainerLog(data.success
      ? `Saved teacher example. Dataset now: ${data.datasetSize}`
      : `Error: ${data.error || 'failed'}`);
  } catch (err) {
    setTrainerLog(`Error: ${err.message}`);
  }
  refreshTrainer();
}

async function refreshTrainer() {
  let data;
  try {
    const res = await fetch('/api/trainer/status');
    data = await res.json();
  } catch (err) {
    document.getElementById('t-status').textContent = 'API OFF';
    document.getElementById('t-status').className = 't-value t-status error';
    return;
  }

  const status = (data.status || 'STOPPED').toUpperCase();
  const step = data.step;
  const loss = data.loss;
  const val_loss = data.val_loss;
  const ds = data.dataset_size || 0;
  const pending = data.pending || 0;

  document.getElementById('t-active-model').textContent = 'SOUL-LLM � ' + (data.active_checkpoint || '?');
  document.getElementById('t-status').textContent = status + (data.worker_alive ? '' : ' (off)');
  document.getElementById('t-status').className = 't-value t-status ' + (status.toLowerCase());
  document.getElementById('t-device').textContent = (data.device || '?') + (data.cuda_available ? ' (CUDA)' : '');
  document.getElementById('t-step').textContent = step === null || step === undefined ? '�' : step;
  document.getElementById('t-loss').textContent =
    loss === null || loss === undefined
      ? (val_loss === null || val_loss === undefined ? '�' : `val ${val_loss}`)
      : `${loss}` + (val_loss === null || val_loss === undefined ? '' : ` / val ${val_loss}`);
  document.getElementById('t-dataset').textContent = `${ds} / ${pending}`;

  // Collection controls (reflect server-side state)
  const modeSel = document.getElementById('t-collect-mode');
  const trainAfterInput = document.getElementById('t-train-after');
  if (data.collectMode && modeSel.value !== data.collectMode) modeSel.value = data.collectMode;
  if (data.trainAfter && trainAfterInput.value !== String(data.trainAfter)) trainAfterInput.value = data.trainAfter;

  // Last evaluation
  if (data.last_eval && data.last_eval.samples) renderTrainerEval(data.last_eval);

  // Checkpoints list
  renderCheckpoints(data.checkpoints || [], data.active_checkpoint);
}

function renderCheckpoints(checkpoints, activeName) {
  const list = document.getElementById('t-checkpoint-list');
  document.getElementById('t-ckpt-count').textContent = checkpoints.length ? `(${checkpoints.length})` : '';
  if (!checkpoints.length) {
    list.innerHTML = '<span class="t-muted">No checkpoints yet. Start training to create one.</span>';
    return;
  }
  list.innerHTML = '';
  checkpoints.slice().sort((a, b) => (b.val_loss ?? b.loss ?? 9) - (a.val_loss ?? a.loss ?? 9)).forEach(ck => {
    const name = ck.name;
    const isActive = name === activeName;
    const row = document.createElement('div');
    row.className = 't-ckpt-row' + (isActive ? ' t-ckpt-active' : '');
    const meta = `${ck.epoch !== undefined ? 'epoch ' + ck.epoch : ''}${ck.step !== undefined ? ' step ' + ck.step : ''}${ck.val_loss !== undefined ? ' loss ' + ck.val_loss : (ck.loss !== undefined ? ' loss ' + ck.loss : '')}${ck.size_mb ? ' � ' + ck.size_mb + 'MB' : ''}`;
    row.innerHTML = `
      <div>
        <div class="t-ckpt-name">${name}${isActive ? ' ?' : ''}</div>
        <div class="t-ckpt-meta">${meta}</div>
      </div>
      <div class="t-ckpt-actions">
        <button class="btn-clean" style="padding:4px 8px;" onclick="trainerUseCheckpoint('${name.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}')">Use</button>
      </div>`;
    list.appendChild(row);
  });
}
