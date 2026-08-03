import sqlite3
import sqlite_vec
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: Path, embedding_dim: int = 768):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS notes (
                note_id         TEXT PRIMARY KEY,
                file_path       TEXT UNIQUE NOT NULL,
                title           TEXT,
                tags            TEXT,
                aliases         TEXT,
                folder          TEXT,
                status          TEXT,
                created_at      TEXT,
                modified_at     TEXT,
                file_hash       TEXT NOT NULL,
                indexed_at      INTEGER,
                chunk_count     INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                note_id         TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                heading_path    TEXT,
                text            TEXT NOT NULL,
                token_count     INTEGER,
                FOREIGN KEY (note_id) REFERENCES notes(note_id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                chunk_id,
                heading_path,
                text,
                tokenize='porter unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id    TEXT PRIMARY KEY,
                note_id     TEXT,
                embedding   float[{self.embedding_dim}]
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);
            CREATE INDEX IF NOT EXISTS idx_notes_folder ON notes(folder);
        """)
        conn.commit()
        logger.info(f"Database initialized with {self.embedding_dim}-dimensional vectors")
        return conn

    def upsert_note(self, note, chunks: List, embeddings: List[List[float]]):
        try:
            self._delete_note_data(note.note_id)

            self.conn.execute("""
                INSERT INTO notes (note_id, file_path, title, tags, aliases, folder, status, 
                                   created_at, modified_at, file_hash, indexed_at, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                note.note_id, 
                note.metadata.file_path, 
                note.metadata.title,
                json.dumps(note.metadata.tags),
                json.dumps(note.metadata.aliases),
                note.metadata.folder,
                note.metadata.status.value if note.metadata.status else None,
                note.metadata.created.isoformat() if note.metadata.created else None,
                note.metadata.modified.isoformat() if note.metadata.modified else None,
                note.metadata.file_hash,
                int(note.metadata.modified.timestamp() if note.metadata.modified else 0),
                len(chunks)
            ))

            for chunk, embedding in zip(chunks, embeddings):
                self.conn.execute("""
                    INSERT INTO chunks (chunk_id, note_id, chunk_index, heading_path, text, token_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk.chunk_id, note.note_id, chunk.index, chunk.heading_path, chunk.text, chunk.token_count))

                self.conn.execute("""
                    INSERT INTO fts_chunks (chunk_id, heading_path, text)
                    VALUES (?, ?, ?)
                """, (chunk.chunk_id, chunk.heading_path, chunk.text))

                vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
                self.conn.execute("""
                    INSERT INTO vec_chunks (chunk_id, note_id, embedding)
                    VALUES (?, ?, ?)
                """, (chunk.chunk_id, note.note_id, vec_bytes))

            self.conn.commit()
            logger.info(f"Indexed note: {note.metadata.file_path} ({len(chunks)} chunks)")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to upsert note {note.note_id}: {e}")
            raise

    def _delete_note_data(self, note_id: str):
        chunk_ids = [row[0] for row in self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE note_id = ?", (note_id,)
        ).fetchall()]
        
        if chunk_ids:
            placeholders = ','.join('?' * len(chunk_ids))
            self.conn.execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
            self.conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", chunk_ids)

        self.conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
        self.conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))

    def get_note_hash(self, file_path: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT file_hash FROM notes WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def query_fts5(self, query: str, limit: int) -> List[str]:
        rows = self.conn.execute("""
            SELECT chunk_id 
            FROM fts_chunks 
            WHERE fts_chunks MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
        return [row[0] for row in rows]

    def query_vectors(self, query_embedding: List[float], limit: int) -> List[str]:
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        rows = self.conn.execute("""
            SELECT chunk_id
            FROM vec_chunks
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (query_bytes, limit)).fetchall()
        return [row[0] for row in rows]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []

        placeholders = ','.join('?' * len(chunk_ids))
        rows = self.conn.execute(f"""
            SELECT 
                c.chunk_id, c.text, c.heading_path,
                n.title, n.file_path, n.tags
            FROM chunks c
            JOIN notes n ON c.note_id = n.note_id
            WHERE c.chunk_id IN ({placeholders})
        """, chunk_ids).fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "text": row[1],
                "heading_path": row[2],
                "note_title": row[3],
                "note_path": row[4],
                "tags": json.loads(row[5]) if row[5] else []
            })
        return results

    def get_all_file_paths(self) -> set:
        rows = self.conn.execute("SELECT file_path FROM notes").fetchall()
        return {row[0] for row in rows}


    def get_all_file_paths(self) -> set[str]:
        """
        Returns a set of all file paths currently in the notes table.
        Used by the Pipeline to detect and clean up deleted files.
        """
        rows = self.conn.execute("SELECT file_path FROM notes").fetchall()
        return {row[0] for row in rows}
