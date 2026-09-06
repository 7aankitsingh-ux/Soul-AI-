const fs = require('fs');
const path = require('path');
const { exec, execSync } = require('child_process');
const os = require('os');
const { ragEngine } = require('./rag');

// Serverless (Vercel) has a read-only filesystem — degrade gracefully.
const IS_SERVERLESS = !!process.env.VERCEL;

// Sandboxed workspace directory
const WORKSPACE_DIR = path.resolve(__dirname, '..', 'workspace');
if (!IS_SERVERLESS && !fs.existsSync(WORKSPACE_DIR)) {
  fs.mkdirSync(WORKSPACE_DIR, { recursive: true });
}

// Memory file
const MEMORY_FILE = path.resolve(__dirname, '..', 'storage', 'memory.json');
const STORAGE_DIR = path.resolve(__dirname, '..', 'storage');
if (!IS_SERVERLESS && !fs.existsSync(STORAGE_DIR)) {
  fs.mkdirSync(STORAGE_DIR, { recursive: true });
}
if (!IS_SERVERLESS && !fs.existsSync(MEMORY_FILE)) {
  fs.writeFileSync(MEMORY_FILE, JSON.stringify({}, null, 2));
}

function resolveWorkspacePath(targetPath) {
  if (!targetPath) return WORKSPACE_DIR;
  if (path.isAbsolute(targetPath)) {
    return targetPath;
  }
  return path.resolve(WORKSPACE_DIR, targetPath);
}

// Registered Tools
const tools = [
  {
    name: 'readFile',
    description: 'Read the text content of a file in the workspace or system.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Relative path in workspace or absolute path' }
      },
      required: ['path']
    },
    handler: async ({ path: targetPath }) => {
      try {
        const fullPath = resolveWorkspacePath(targetPath);
        if (!fs.existsSync(fullPath)) {
          return { error: `File not found: ${targetPath}` };
        }
        const stat = fs.statSync(fullPath);
        if (stat.isDirectory()) {
          return { error: `Target is a directory, not a file: ${targetPath}` };
        }
        const content = fs.readFileSync(fullPath, 'utf8');
        return {
          path: fullPath,
          size: stat.size,
          content: content.length > 10000 ? content.slice(0, 10000) + '\n... [truncated]' : content
        };
      } catch (err) {
        return { error: err.message };
      }
    }
  },
  {
    name: 'writeFile',
    description: 'Create or overwrite a file in the workspace with given content.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Relative file path inside workspace' },
        content: { type: 'string', description: 'The text or code content to write' }
      },
      required: ['path', 'content']
    },
    handler: async ({ path: targetPath, content }) => {
      try {
        const fullPath = resolveWorkspacePath(targetPath);
        const dir = path.dirname(fullPath);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(fullPath, content, 'utf8');
        return {
          success: true,
          message: `Successfully wrote ${content.length} characters to ${path.basename(fullPath)}`,
          path: fullPath
        };
      } catch (err) {
        return { error: err.message };
      }
    }
  },
  {
    name: 'listFiles',
    description: 'List files and folders in the workspace or a given directory.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Directory path (optional, defaults to workspace root)' }
      }
    },
    handler: async ({ path: targetPath = '' }) => {
      try {
        const fullPath = resolveWorkspacePath(targetPath);
        if (!fs.existsSync(fullPath)) {
          return { error: `Directory not found: ${targetPath}` };
        }
        const entries = fs.readdirSync(fullPath, { withFileTypes: true });
        const list = entries.map(e => ({
          name: e.name,
          type: e.isDirectory() ? 'directory' : 'file',
          size: e.isFile() ? fs.statSync(path.join(fullPath, e.name)).size : null
        }));
        return { directory: fullPath, items: list };
      } catch (err) {
        return { error: err.message };
      }
    }
  },
  {
    name: 'searchFiles',
    description: 'Search for files by filename pattern or search file contents within the workspace.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Filename substring or text to search for' },
        contentSearch: { type: 'boolean', description: 'If true, searches file contents; otherwise searches filenames' }
      },
      required: ['query']
    },
    handler: async ({ query, contentSearch = false }) => {
      try {
        const results = [];
        function scan(dir) {
          const items = fs.readdirSync(dir, { withFileTypes: true });
          for (const item of items) {
            const itemPath = path.join(dir, item.name);
            if (item.isDirectory()) {
              if (item.name !== 'node_modules' && item.name !== '.git') {
                scan(itemPath);
              }
            } else {
              if (!contentSearch && item.name.toLowerCase().includes(query.toLowerCase())) {
                results.push({ path: path.relative(WORKSPACE_DIR, itemPath), matchType: 'filename' });
              } else if (contentSearch) {
                try {
                  const content = fs.readFileSync(itemPath, 'utf8');
                  if (content.toLowerCase().includes(query.toLowerCase())) {
                    results.push({ path: path.relative(WORKSPACE_DIR, itemPath), matchType: 'content' });
                  }
                } catch {
                  // ignore binary files
                }
              }
            }
          }
        }
        scan(WORKSPACE_DIR);
        return { query, matchesFound: results.length, matches: results.slice(0, 20) };
      } catch (err) {
        return { error: err.message };
      }
    }
  },
  {
    name: 'runCommand',
    description: 'Execute a terminal command (PowerShell / Command Prompt) in the workspace.',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'The shell command to execute' },
        timeoutMs: { type: 'number', description: 'Timeout in ms (default 15000)' }
      },
      required: ['command']
    },
    handler: async ({ command, timeoutMs = 15000 }) => {
      return new Promise((resolve) => {
        exec(command, { cwd: WORKSPACE_DIR, timeout: timeoutMs, shell: 'powershell.exe' }, (error, stdout, stderr) => {
          if (error) {
            resolve({
              success: false,
              exitCode: error.code || 1,
              error: error.message,
              stdout: stdout ? stdout.trim() : '',
              stderr: stderr ? stderr.trim() : ''
            });
          } else {
            resolve({
              success: true,
              exitCode: 0,
              stdout: stdout ? stdout.trim() : '(No output)',
              stderr: stderr ? stderr.trim() : ''
            });
          }
        });
      });
    }
  },
  {
    name: 'calculate',
    description: 'Safely evaluate a mathematical calculation or formula.',
    parameters: {
      type: 'object',
      properties: {
        expression: { type: 'string', description: 'Mathematical expression (e.g. "(128 * 45) + 500" or "Math.sqrt(144)")' }
      },
      required: ['expression']
    },
    handler: async ({ expression }) => {
      try {
        // Sanitize to allow only standard math characters and Math methods
        const sanitized = expression.replace(/[^0-9+\-*/().,%^ Math\.sqrtcossinabstandegceilfloorpowminmaxEPI]/g, '');
        // Evaluate using safe Function scope
        const fn = new Function(`"use strict"; return (${sanitized});`);
        const result = fn();
        return { expression, result: Number(result) };
      } catch (err) {
        return { error: `Calculation failed: ${err.message}` };
      }
    }
  },
  {
    name: 'getSystemInfo',
    description: 'Retrieve system telemetry: OS, CPU, RAM, and NVIDIA GPU statistics.',
    parameters: {
      type: 'object',
      properties: {}
    },
    handler: async () => {
      try {
        const totalMemGb = (os.totalmem() / (1024 ** 3)).toFixed(2);
        const freeMemGb = (os.freemem() / (1024 ** 3)).toFixed(2);
        const usedMemGb = (totalMemGb - freeMemGb).toFixed(2);
        const cpus = os.cpus();
        
        let gpuInfo = 'NVIDIA GPU telemetry not available';
        try {
          const nvidiaOut = execSync('nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader', { encoding: 'utf8' });
          gpuInfo = nvidiaOut.trim();
        } catch {
          // fallback if nvidia-smi fails
        }

        return {
          platform: os.platform(),
          release: os.release(),
          cpuModel: cpus[0]?.model || 'Unknown',
          cpuCores: cpus.length,
          ram: {
            totalGb: `${totalMemGb} GB`,
            usedGb: `${usedMemGb} GB`,
            freeGb: `${freeMemGb} GB`,
            percentUsed: `${Math.round((usedMemGb / totalMemGb) * 100)}%`
          },
          gpu: gpuInfo,
          workspacePath: WORKSPACE_DIR
        };
      } catch (err) {
        return { error: err.message };
      }
    }
  },
  {
    name: 'localRagSearch',
    description: 'Search offline indexed documents and notes in the local knowledge base.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search term or question to find in offline documents' },
        topK: { type: 'number', description: 'Number of top relevant excerpts to return (default 3)' }
      },
      required: ['query']
    },
    handler: async ({ query, topK = 3 }) => {
      try {
        return ragEngine.search(query, topK);
      } catch (err) {
        return { error: err.message };
      }
    }
  },
  {
    name: 'storeMemory',
    description: 'Save a key-value or knowledge fact to persistent offline memory.',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Unique identifier or category' },
        value: { type: 'string', description: 'Information to remember permanently' }
      },
      required: ['key', 'value']
    },
    handler: async ({ key, value }) => {
      try {
        let mem = {};
        if (fs.existsSync(MEMORY_FILE)) {
          mem = JSON.parse(fs.readFileSync(MEMORY_FILE, 'utf8') || '{}');
        }
        mem[key] = {
          value,
          updatedAt: new Date().toISOString()
        };
        fs.writeFileSync(MEMORY_FILE, JSON.stringify(mem, null, 2));
        return { success: true, message: `Memory saved for key '${key}'` };
      } catch (err) {
        return { error: err.message };
      }
    }
  },
  {
    name: 'retrieveMemory',
    description: 'Look up facts or values stored in long-term offline memory.',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'Key name or leave blank to retrieve all stored memory' }
      }
    },
    handler: async ({ key = '' }) => {
      try {
        if (!fs.existsSync(MEMORY_FILE)) {
          return { memory: {} };
        }
        const mem = JSON.parse(fs.readFileSync(MEMORY_FILE, 'utf8') || '{}');
        if (key && mem[key]) {
          return { key, data: mem[key] };
        }
        return { memory: mem };
      } catch (err) {
        return { error: err.message };
      }
    }
  }
];

function getToolDefinitions() {
  return tools.map(t => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters
    }
  }));
}

async function executeTool(name, args) {
  const tool = tools.find(t => t.name === name);
  if (!tool) {
    return { error: `Tool '${name}' is not registered.` };
  }
  try {
    const result = await tool.handler(args || {});
    return result;
  } catch (err) {
    return { error: `Error executing ${name}: ${err.message}` };
  }
}

module.exports = {
  tools,
  getToolDefinitions,
  executeTool,
  WORKSPACE_DIR,
  resolveWorkspacePath
};
