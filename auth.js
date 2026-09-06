// Local user accounts, sessions, and chat history persisted to storage/
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Serverless (Vercel) environments have a read-only filesystem. When running
// there we degrade to an ephemeral in-memory store instead of crashing on
// mkdir/write. Persistence is lost between requests — fine for a demo shell.
const IS_SERVERLESS = !!process.env.VERCEL;

const STORAGE_DIR = path.join(__dirname, 'storage');
const USERS_FILE = path.join(STORAGE_DIR, 'users.json');
const SESSIONS_FILE = path.join(STORAGE_DIR, 'sessions.json');
const CHATS_DIR = path.join(STORAGE_DIR, 'chats');

function ensureStorage() {
  if (IS_SERVERLESS) return;
  fs.mkdirSync(STORAGE_DIR, { recursive: true });
  fs.mkdirSync(CHATS_DIR, { recursive: true });
  if (!fs.existsSync(USERS_FILE)) fs.writeFileSync(USERS_FILE, '{}', 'utf8');
  if (!fs.existsSync(SESSIONS_FILE)) fs.writeFileSync(SESSIONS_FILE, '{}', 'utf8');
}

function writeJson(file, data) {
  if (IS_SERVERLESS) return;
  fs.writeFileSync(file, JSON.stringify(data, null, 2), 'utf8');
}

// ─── Password hashing (scrypt, built into Node — no extra deps) ───
function hashPassword(password, salt = crypto.randomBytes(16).toString('hex')) {
  const hash = crypto.scryptSync(String(password), salt, 64).toString('hex');
  return { salt, hash };
}

function verifyPassword(password, salt, expected) {
  try {
    const { hash } = hashPassword(password, salt);
    return crypto.timingSafeEqual(Buffer.from(hash, 'hex'), Buffer.from(expected, 'hex'));
  } catch {
    return false;
  }
}

// ─── Sessions (simple opaque tokens) ───
function createSession(username) {
  ensureStorage();
  const token = crypto.randomBytes(24).toString('hex');
  const sessions = readJson(SESSIONS_FILE, {});
  sessions[token] = { username, createdAt: Date.now() };
  writeJson(SESSIONS_FILE, sessions);
  return token;
}

function destroySession(token) {
  const sessions = readJson(SESSIONS_FILE, {});
  delete sessions[token];
  writeJson(SESSIONS_FILE, sessions);
}

function usernameFromToken(token) {
  if (!token) return null;
  const sessions = readJson(SESSIONS_FILE, {});
  const session = sessions[token];
  return session ? session.username : null;
}

// ─── Users ───
function registerUser(username, password) {
  ensureStorage();
  const users = readJson(USERS_FILE, {});
  if (users[username]) return { error: 'Username already taken' };
  const { salt, hash } = hashPassword(password);
  users[username] = { username, salt, hash, createdAt: Date.now() };
  writeJson(USERS_FILE, users);
  return { user: { username } };
}

function loginUser(username, password) {
  const users = readJson(USERS_FILE, {});
  const user = users[username];
  if (!user || !verifyPassword(password, user.salt, user.hash)) {
    return { error: 'Invalid username or password' };
  }
  return { user: { username } };
}

function sanitizeUsername(username) {
  return typeof username === 'string' && /^[a-zA-Z0-9_]{3,20}$/.test(username) ? username : null;
}

// ─── Chat history ───
function chatsFile(username) {
  return path.join(CHATS_DIR, `${username}.json`);
}

function getChats(username) {
  const file = chatsFile(username);
  if (!fs.existsSync(file)) return [];
  const data = readJson(file, { chats: [] });
  return data.chats || [];
}

function saveChat(username, chat) {
  if (!chat || !chat.id) return { error: 'chat id required' };
  const file = chatsFile(username);
  const chats = getChats(username);
  const idx = chats.findIndex(c => c.id === chat.id);
  if (idx >= 0) {
    chats[idx] = chat;
  } else {
    chats.push(chat);
  }
  writeJson(file, { chats });
  return chat;
}

function deleteChat(username, id) {
  const file = chatsFile(username);
  const chats = getChats(username).filter(c => c.id !== id);
  writeJson(file, { chats });
}

// Cookie helpers
function parseCookies(req) {
  const header = req.headers.cookie || '';
  const cookies = {};
  header.split(';').forEach(part => {
    const eq = part.indexOf('=');
    if (eq > 0) {
      cookies[part.slice(0, eq).trim()] = decodeURIComponent(part.slice(eq + 1).trim());
    }
  });
  return cookies;
}

module.exports = {
  ensureStorage,
  registerUser,
  loginUser,
  sanitizeUsername,
  createSession,
  destroySession,
  usernameFromToken,
  getChats,
  saveChat,
  deleteChat,
  parseCookies
};