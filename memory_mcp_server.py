#!/usr/bin/env python3
"""
StoryGen Memory MCP Server
Lightweight persistent memory for StoryGen agent system.
Uses SQLite FTS5 for storage and search - zero ML dependencies.
Designed for opencode MCP integration.
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

MEMORY_DB = os.environ.get("STORYGEN_MEMORY_DB", str(Path.home() / ".storygen_memory.db"))
DEFAULT_USER = os.environ.get("STORYGEN_MEMORY_USER", "storygen-agent")

mcp = FastMCP("StoryGen Memory")


def _get_db() -> sqlite3.Connection:
    """Get thread-safe SQLite connection with FTS5."""
    conn = sqlite3.connect(MEMORY_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            tags TEXT NOT NULL DEFAULT '[]',
            user_id TEXT NOT NULL DEFAULT 'storygen-agent',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
        CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
        CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            key, value, category, tags, content='memories', content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, key, value, category, tags)
            VALUES (new.id, new.key, new.value, new.category, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, key, value, category, tags)
            VALUES ('delete', old.id, old.key, old.value, old.category, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, key, value, category, tags)
            VALUES ('delete', old.id, old.key, old.value, old.category, old.tags);
            INSERT INTO memories_fts(rowid, key, value, category, tags)
            VALUES (new.id, new.key, new.value, new.category, new.tags);
        END;
    """)


@mcp.tool()
def add_memory(
    key: str,
    value: str,
    category: str = "general",
    tags: Optional[list[str]] = None,
    user_id: Optional[str] = None,
) -> str:
    """Store a memory with key-value pair and category for later retrieval.
    
    Args:
        key: Short identifier for the memory (e.g. "sca_q_unchanged")
        value: Full content/description of the memory
        category: Grouping category (e.g. "architecture", "experiment", "code_location", "failure_pattern")
        tags: Optional list of searchable tags
        user_id: Optional user/agent identifier
    Returns:
        JSON with id and confirmation
    """
    now = time.time()
    uid = user_id or DEFAULT_USER
    tags_json = json.dumps(tags or [])
    
    conn = _get_db()
    try:
        cur = conn.execute(
            "INSERT INTO memories (key, value, category, tags, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, value, category, tags_json, uid, now, now),
        )
        conn.commit()
        mem_id = cur.lastrowid
        return json.dumps({"id": mem_id, "key": key, "status": "stored"})
    finally:
        conn.close()


@mcp.tool()
def search_memories(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    user_id: Optional[str] = None,
) -> str:
    """Search memories by semantic content using full-text search.
    
    Args:
        query: Search text (searches key, value, category, tags)
        category: Optional filter by category
        limit: Max results to return
        user_id: Optional user filter
    Returns:
        JSON array of matching memories
    """
    uid = user_id or DEFAULT_USER
    conn = _get_db()
    try:
        if category:
            rows = conn.execute(
                """SELECT m.id, m.key, m.value, m.category, m.tags, m.created_at, m.updated_at
                   FROM memories_fts f JOIN memories m ON f.rowid = m.id
                   WHERE memories_fts MATCH ? AND m.category = ? AND m.user_id = ?
                   ORDER BY rank LIMIT ?""",
                (query, category, uid, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT m.id, m.key, m.value, m.category, m.tags, m.created_at, m.updated_at
                   FROM memories_fts f JOIN memories m ON f.rowid = m.id
                   WHERE memories_fts MATCH ? AND m.user_id = ?
                   ORDER BY rank LIMIT ?""",
                (query, uid, limit),
            ).fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "key": r["key"],
                "value": r["value"],
                "category": r["category"],
                "tags": json.loads(r["tags"]),
                "created_at": datetime.fromtimestamp(r["created_at"]).isoformat(),
            })
        return json.dumps(results, ensure_ascii=False)
    finally:
        conn.close()


@mcp.tool()
def get_memory(memory_id: int) -> str:
    """Retrieve a specific memory by ID.
    
    Args:
        memory_id: The memory ID to retrieve
    Returns:
        JSON with memory details or error
    """
    conn = _get_db()
    try:
        r = conn.execute(
            "SELECT id, key, value, category, tags, created_at, updated_at FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if r is None:
            return json.dumps({"error": f"Memory {memory_id} not found"})
        return json.dumps({
            "id": r["id"],
            "key": r["key"],
            "value": r["value"],
            "category": r["category"],
            "tags": json.loads(r["tags"]),
            "created_at": datetime.fromtimestamp(r["created_at"]).isoformat(),
        }, ensure_ascii=False)
    finally:
        conn.close()


@mcp.tool()
def delete_memory(memory_id: int) -> str:
    """Delete a memory by ID.
    
    Args:
        memory_id: The memory ID to delete
    Returns:
        JSON confirmation
    """
    conn = _get_db()
    try:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return json.dumps({"status": "deleted", "id": memory_id})
    finally:
        conn.close()


@mcp.tool()
def list_categories(user_id: Optional[str] = None) -> str:
    """List all memory categories with counts.
    
    Args:
        user_id: Optional user filter
    Returns:
        JSON array of {category, count}
    """
    uid = user_id or DEFAULT_USER
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT category, COUNT(*) as count FROM memories WHERE user_id = ? GROUP BY category ORDER BY count DESC",
            (uid,),
        ).fetchall()
        results = [{"category": r["category"], "count": r["count"]} for r in rows]
        return json.dumps(results)
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run(transport="stdio")
