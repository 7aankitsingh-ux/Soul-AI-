const fs = require('fs');
const path = require('path');

const IS_SERVERLESS = !!process.env.VERCEL;
const KNOWLEDGE_DIR = path.resolve(__dirname, '..', 'storage', 'knowledge');
const INDEX_FILE = path.resolve(__dirname, '..', 'storage', 'rag_index.json');

if (!IS_SERVERLESS && !fs.existsSync(KNOWLEDGE_DIR)) {
  fs.mkdirSync(KNOWLEDGE_DIR, { recursive: true });
}

// Simple fast tokenizer for offline ranking
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 1);
}

// Split document text into chunks with overlap
function chunkText(text, chunkSize = 400, overlap = 80) {
  const words = text.split(/\s+/);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const chunkWords = words.slice(i, i + chunkSize);
    chunks.push(chunkWords.join(' '));
    i += chunkSize - overlap;
  }
  return chunks.length ? chunks : [text];
}

class OfflineRagEngine {
  constructor() {
    this.documents = [];
    this.chunks = [];
    this.loadIndex();
  }

  loadIndex() {
    try {
      if (fs.existsSync(INDEX_FILE)) {
        const data = JSON.parse(fs.readFileSync(INDEX_FILE, 'utf8'));
        this.documents = data.documents || [];
        this.chunks = data.chunks || [];
      } else {
        this.rebuildFromFolder();
      }
    } catch {
      this.documents = [];
      this.chunks = [];
    }
  }

  saveIndex() {
    try {
      fs.writeFileSync(INDEX_FILE, JSON.stringify({
        documents: this.documents,
        chunks: this.chunks
      }, null, 2));
    } catch (err) {
      console.error('Failed to save RAG index:', err);
    }
  }

  rebuildFromFolder() {
    this.documents = [];
    this.chunks = [];
    if (!fs.existsSync(KNOWLEDGE_DIR)) return;

    const files = fs.readdirSync(KNOWLEDGE_DIR);
    for (const file of files) {
      const fullPath = path.join(KNOWLEDGE_DIR, file);
      const stat = fs.statSync(fullPath);
      if (stat.isFile()) {
        const content = fs.readFileSync(fullPath, 'utf8');
        this.addDocument(file, content, false);
      }
    }
    this.saveIndex();
  }

  addDocument(filename, content, save = true) {
    // Remove if already exists
    this.documents = this.documents.filter(d => d.filename !== filename);
    this.chunks = this.chunks.filter(c => c.source !== filename);

    const docMeta = {
      filename,
      charCount: content.length,
      indexedAt: new Date().toISOString()
    };
    this.documents.push(docMeta);

    // Chunk content
    const textChunks = chunkText(content);
    for (let idx = 0; idx < textChunks.length; idx++) {
      const chunkTextStr = textChunks[idx];
      const tokens = tokenize(chunkTextStr);
      const tf = {};
      for (const token of tokens) {
        tf[token] = (tf[token] || 0) + 1;
      }
      this.chunks.push({
        id: `${filename}_chunk_${idx}`,
        source: filename,
        chunkIndex: idx,
        text: chunkTextStr,
        tokensCount: tokens.length,
        tf
      });
    }

    if (save) {
      this.saveIndex();
    }
  }

  removeDocument(filename) {
    this.documents = this.documents.filter(d => d.filename !== filename);
    this.chunks = this.chunks.filter(c => c.source !== filename);
    const fullPath = path.join(KNOWLEDGE_DIR, filename);
    if (fs.existsSync(fullPath)) {
      fs.unlinkSync(fullPath);
    }
    this.saveIndex();
    return { success: true, removed: filename };
  }

  search(query, topK = 3) {
    if (!this.chunks.length) {
      return {
        query,
        count: 0,
        results: [],
        message: 'No documents in knowledge base. Add files in storage/knowledge or via dashboard.'
      };
    }

    const queryTokens = tokenize(query);
    if (!queryTokens.length) {
      return { query, count: 0, results: [] };
    }

    // BM25-style scoring
    const N = this.chunks.length;
    const avgDocLen = this.chunks.reduce((acc, c) => acc + c.tokensCount, 0) / N;
    const k1 = 1.5;
    const b = 0.75;

    const scores = this.chunks.map(chunk => {
      let score = 0;
      const docLen = chunk.tokensCount;

      for (const token of queryTokens) {
        // Document frequency
        const df = this.chunks.filter(c => c.tf[token]).length;
        if (df > 0) {
          const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
          const f = chunk.tf[token] || 0;
          const termScore = idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (docLen / avgDocLen))));
          score += termScore;
        }
      }

      return {
        score,
        source: chunk.source,
        chunkIndex: chunk.chunkIndex,
        text: chunk.text
      };
    });

    const ranked = scores
      .filter(s => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return {
      query,
      count: ranked.length,
      results: ranked
    };
  }

  getStats() {
    return {
      totalDocuments: this.documents.length,
      totalChunks: this.chunks.length,
      documents: this.documents
    };
  }
}

const ragEngine = new OfflineRagEngine();

module.exports = {
  ragEngine,
  KNOWLEDGE_DIR
};
