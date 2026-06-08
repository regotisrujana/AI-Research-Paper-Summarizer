import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import DATA_DIR, DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT 'default',
                paper_id INTEGER,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS paper_summaries (
                paper_id INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                paper_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                paper_name TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                is_reference INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (paper_id, chunk_id),
                FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
            );
            """
        )
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)")]
        if "session_id" not in columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_paper(name: str, file_path: Path, page_count: int, chunk_count: int) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO papers (name, file_path, page_count, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, str(file_path), page_count, chunk_count, utc_now()),
        )
        return int(cursor.lastrowid)


def update_paper_chunks(paper_id: int, chunk_count: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE papers SET chunk_count = ? WHERE id = ?",
            (chunk_count, paper_id),
        )


def list_papers() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return list(conn.execute("SELECT * FROM papers ORDER BY created_at DESC"))


def get_paper(paper_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()


def get_paper_by_name(name: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM papers WHERE name = ?", (name,)).fetchone()


def add_chat_message(session_id: str, paper_id: int | None, role: str, content: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (session_id, paper_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, paper_id, role, content, utc_now()),
        )


def list_chat_history(session_id: str | None = None, paper_id: int | None = None) -> list[sqlite3.Row]:
    with get_connection() as conn:
        if session_id is None and paper_id is None:
            return list(conn.execute("SELECT * FROM chat_messages ORDER BY created_at ASC"))
        if session_id is not None and paper_id is None:
            return list(
                conn.execute(
                    "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                )
            )
        if session_id is None and paper_id is not None:
            return list(
                conn.execute(
                    "SELECT * FROM chat_messages WHERE paper_id = ? ORDER BY created_at ASC",
                    (paper_id,),
                )
            )
        return list(
            conn.execute(
                """
                SELECT * FROM chat_messages
                WHERE session_id = ? AND paper_id = ?
                ORDER BY created_at ASC
                """,
                (session_id, paper_id),
            )
        )


def delete_chat_history(paper_id: int, session_id: str | None = None) -> None:
    with get_connection() as conn:
        if session_id:
            conn.execute("DELETE FROM chat_messages WHERE paper_id = ? AND session_id = ?", (paper_id, session_id))
        else:
            conn.execute("DELETE FROM chat_messages WHERE paper_id = ?", (paper_id,))


def get_latest_session_paper_id(session_id: str) -> int | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT paper_id FROM chat_messages
            WHERE session_id = ? AND paper_id IS NOT NULL
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        return int(row["paper_id"]) if row else None


def get_cached_summary(paper_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT paper_id, summary, created_at, updated_at FROM paper_summaries WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()


def delete_paper_record(paper_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE paper_id = ?", (paper_id,))
        conn.execute("DELETE FROM paper_summaries WHERE paper_id = ?", (paper_id,))
        conn.execute("DELETE FROM chunk_embeddings WHERE paper_id = ?", (paper_id,))
        conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))


def save_summary(paper_id: int, summary: str) -> None:
    now = utc_now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO paper_summaries (paper_id, summary, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (paper_id, summary, now, now),
        )


def replace_paper_embeddings(paper_id: int, rows: list[dict]) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM chunk_embeddings WHERE paper_id = ?", (paper_id,))
        conn.executemany(
            """
            INSERT INTO chunk_embeddings
            (paper_id, chunk_id, paper_name, page_number, text, embedding_json, is_reference)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    paper_id,
                    row["chunk_id"],
                    row["paper_name"],
                    row["page_number"],
                    row["text"],
                    json.dumps(row["embedding"]),
                    1 if row.get("is_reference") else 0,
                )
                for row in rows
            ],
        )


def count_paper_embeddings(paper_id: int) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM chunk_embeddings WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        return int(row["count"]) if row else 0


def get_paper_embeddings(paper_id: int | None = None) -> list[dict]:
    with get_connection() as conn:
        if paper_id is None:
            rows = conn.execute("SELECT * FROM chunk_embeddings").fetchall()
        else:
            rows = conn.execute("SELECT * FROM chunk_embeddings WHERE paper_id = ?", (paper_id,)).fetchall()
    return [
        {
            "paper_id": int(row["paper_id"]),
            "chunk_id": str(row["chunk_id"]),
            "paper_name": str(row["paper_name"]),
            "page_number": int(row["page_number"]),
            "text": str(row["text"]),
            "embedding": json.loads(str(row["embedding_json"])),
            "is_reference": bool(row["is_reference"]),
        }
        for row in rows
    ]


def delete_paper_embeddings(paper_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM chunk_embeddings WHERE paper_id = ?", (paper_id,))
