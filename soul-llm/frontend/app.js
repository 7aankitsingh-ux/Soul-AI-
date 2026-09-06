// SOUL-LLM — Simple Black & White Chat

let conversationHistory = [];

function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });

  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');
  html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  const parts = html.split(/(<pre>[\s\S]*?<\/pre>)/);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  }

  return parts.join('');
}

const messages = document.getElementById('messages');
const welcome = document.getElementById('welcome');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');

document.addEventListener('DOMContentLoaded', () => {
  sendBtn.addEventListener('click', handleSendMessage);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });

  checkStatus();
});

async function checkStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'healthy' && data.model_loaded) {
      statusEl.textContent = 'Ready';
      statusEl.style.color = '#ffffff';
    } else {
      statusEl.textContent = 'Offline';
      statusEl.style.color = '#606060';
    }
  } catch {
    statusEl.textContent = 'Offline';
    statusEl.style.color = '#606060';
  }
}

window.sendChip = (text) => {
  input.value = text;
  handleSendMessage();
};

async function handleSendMessage() {
  const text = input.value.trim();
  if (!text) return;

  if (welcome && welcome.parentNode) {
    welcome.style.display = 'none';
  }

  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;

  appendUserMessage(text);
  conversationHistory.push({ role: 'user', content: text });

  const assistantBubble = createAssistantBubble();

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'soul-llm',
        messages: conversationHistory,
        temperature: 0.7,
        max_tokens: 150,
        stream: false
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || '(empty response)';

    assistantBubble.innerHTML = renderMarkdown(content);
    conversationHistory.push({ role: 'assistant', content });

  } catch (err) {
    assistantBubble.innerHTML = renderMarkdown(`**Error:** ${err.message}. Please check if the local server is running.`);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="msg-avatar user">You</div>
    <div class="msg-main">
      <div class="msg-bubble">${renderMarkdown(text)}</div>
    </div>
  `;
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
}

function createAssistantBubble() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';

  const mainDiv = document.createElement('div');
  mainDiv.className = 'msg-main';

  const bubbleDiv = document.createElement('div');
  bubbleDiv.className = 'msg-bubble';
  bubbleDiv.innerHTML = '<span class="thinking">Thinking...</span>';

  mainDiv.appendChild(bubbleDiv);
  row.innerHTML = '<div class="msg-avatar ai">AI</div>';
  row.appendChild(mainDiv);

  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;

  return bubbleDiv;
}
