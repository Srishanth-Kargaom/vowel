"""
memory.py — Persistent memory for Vowel AI Code Assistant.

Two-layer memory:
  Short-term : session state (current conversation, in RAM)
  Long-term  : SQLite for text + FAISS for semantic search across all past conversations

NOTE: On Streamlit Cloud, SQLite/FAISS files are written to the app's working
directory. They persist for the lifetime of the deployment but reset on reboot.
For true cross-session persistence, swap SQLite for a cloud DB (e.g. Supabase). 
"""

import sqlite3
import faiss
import numpy as np
import pickle
import os
from datetime import datetime

DB_PATH   = "chat_history.db"
IDX_PATH  = "memory_index.bin"
META_PATH = "memory_meta.pkl"
DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

# ── DATABASE SETUP ─────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL,
            timestamp TEXT    NOT NULL,
            session   TEXT    NOT NULL
        )
    """)
    con.commit()
    con.close()

# ── FAISS MEMORY INDEX ─────────────────────────────────────────────────────────
def load_memory_index():
    if os.path.exists(IDX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(IDX_PATH)
        # Guard: if the saved index has the wrong dimension, discard and rebuild
        if index.d != DIM:
            index = faiss.IndexFlatIP(DIM)
            meta = []
            return index, meta
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(DIM)
        meta = []
    return index, meta

def save_memory_index(index, meta):
    faiss.write_index(index, IDX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

# ── CORE MEMORY CLASS ──────────────────────────────────────────────────────────
class MemoryManager:
    """
    Manages short-term (session) and long-term (persistent) memory.

    Short-term: plain list in RAM — current session only.
    Long-term : SQLite stores all text, FAISS stores embeddings for semantic search.
    """

    def __init__(self, embed_fn, session_id: str = None, max_short: int = 10):
        init_db()
        self.embed_fn   = embed_fn
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_short  = max_short
        self.short_term = []
        self.index, self.meta = load_memory_index()

    # ── SHORT-TERM ─────────────────────────────────────────────────────────────
    def add(self, role: str, content: str):
        """Save one turn to short-term + long-term memory."""
        ts = datetime.now().isoformat()

        # Short-term (RAM)
        self.short_term.append({"role": role, "content": content, "ts": ts})
        if len(self.short_term) > self.max_short * 2:
            self.short_term = self.short_term[-(self.max_short * 2):]

        # Long-term — SQLite
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO conversations (role, content, timestamp, session) VALUES (?,?,?,?)",
            (role, content, ts, self.session_id),
        )
        con.commit()
        con.close()

        # Long-term — FAISS
        vec = self.embed_fn(content).astype("float32")
        self.index.add(vec.reshape(1, -1))
        self.meta.append({
            "role": role, "content": content,
            "ts": ts, "session": self.session_id,
        })
        save_memory_index(self.index, self.meta)

    def get_short_term(self):
        return self.short_term

    def format_short_term(self) -> str:
        if not self.short_term:
            return ""
        lines = ["\n--- Recent conversation ---"]
        for t in self.short_term[-6:]:
            who = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{who}: {t['content'][:300]}")
        return "\n".join(lines)

    # ── LONG-TERM SEMANTIC SEARCH ──────────────────────────────────────────────
    def search_long_term(self, query: str, top_k: int = 3) -> list:
        if self.index.ntotal == 0:
            return []
        q_vec = self.embed_fn(query).astype("float32").reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and float(score) > 0.5:
                item = self.meta[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    def format_long_term(self, query: str) -> str:
        hits = self.search_long_term(query, top_k=3)
        if not hits:
            return ""
        lines = ["\n--- Relevant past context ---"]
        for h in hits:
            who = "User" if h["role"] == "user" else "Assistant"
            lines.append(f"{who} (earlier): {h['content'][:200]}")
        return "\n".join(lines)

    def used_long_term(self, query: str) -> bool:
        return len(self.search_long_term(query, top_k=1)) > 0

    # ── HISTORY MANAGEMENT ─────────────────────────────────────────────────────
    def get_all_history(self) -> list:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT id, role, content, timestamp, session FROM conversations ORDER BY id ASC"
        ).fetchall()
        con.close()
        return [{"id": r[0], "role": r[1], "content": r[2],
                 "ts": r[3], "session": r[4]} for r in rows]

    def delete_last(self):
        con = sqlite3.connect(DB_PATH)
        ids = con.execute(
            "SELECT id FROM conversations ORDER BY id DESC LIMIT 2"
        ).fetchall()
        for (rid,) in ids:
            con.execute("DELETE FROM conversations WHERE id=?", (rid,))
        con.commit()
        con.close()
        if len(self.short_term) >= 2:
            self.short_term = self.short_term[:-2]
        self._rebuild_index()

    def clear_all(self):
        con = sqlite3.connect(DB_PATH)
        con.execute("DELETE FROM conversations")
        con.commit()
        con.close()
        self.short_term = []
        self.index = faiss.IndexFlatIP(DIM)
        self.meta  = []
        save_memory_index(self.index, self.meta)

    def _rebuild_index(self):
        rows = self.get_all_history()
        self.index = faiss.IndexFlatIP(DIM)
        self.meta  = []
        for row in rows:
            vec = self.embed_fn(row["content"]).astype("float32")
            self.index.add(vec.reshape(1, -1))
            self.meta.append({
                "role": row["role"], "content": row["content"],
                "ts": row["ts"], "session": row["session"],
            })
        save_memory_index(self.index, self.meta)
