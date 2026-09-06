"""
Soul AI RAG Knowledge Engine for SOUL-LLM
==========================================
Lightweight BM25 / token matching knowledge retrieval for offline documents.
"""

import os
import re
import json
import math
from typing import List, Dict, Any

KNOWLEDGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge"))
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)


class SimpleRAG:
    def __init__(self, storage_dir: str = KNOWLEDGE_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.documents: List[Dict[str, Any]] = []
        self._load_documents()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\w+", text)]

    def _load_documents(self):
        self.documents = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.documents.append(data)
                except Exception:
                    pass

    def add_document(self, title: str, content: str) -> Dict[str, Any]:
        """Indexes and saves a document into the offline knowledge base."""
        doc_id = re.sub(r"[^\w\-]", "_", title.lower())
        doc_data = {
            "id": doc_id,
            "title": title,
            "content": content,
            "tokens": self._tokenize(content),
            "size": len(content)
        }
        fpath = os.path.join(self.storage_dir, f"{doc_id}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2)

        # Update in-memory
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        self.documents.append(doc_data)
        return {"status": "success", "id": doc_id, "title": title, "tokensCount": len(doc_data["tokens"])}

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search documents using term frequency ranking."""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scored = []
        for doc in self.documents:
            doc_tokens = doc.get("tokens", [])
            if not doc_tokens:
                continue
            doc_len = len(doc_tokens)
            score = 0.0
            for qt in query_tokens:
                tf = doc_tokens.count(qt)
                if tf > 0:
                    score += (tf / doc_len) * (1.0 + math.log(1 + 10 / max(1, len(self.documents))))
            if score > 0:
                scored.append({
                    "title": doc["title"],
                    "score": round(score, 4),
                    "snippet": doc["content"][:300] + ("..." if len(doc["content"]) > 300 else "")
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {"id": d["id"], "title": d["title"], "size": d["size"], "tokens": len(d.get("tokens", []))}
            for d in self.documents
        ]


rag_engine = SimpleRAG()
