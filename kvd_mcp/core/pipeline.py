# src/kvd_mcp/core/pipeline.py
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from .models import Note, Chunk
from .storage import Storage
from .chunker import Chunker, ChunkerConfig
from .embeddings import Embedder
from .db import Database

logger = logging.getLogger(__name__)


class Pipeline:
    """
    The Write Orchestrator.

    Coordinates the full ingestion flow:
        Storage (read file)
            → Chunker (split into sections)
                → Embedder (generate vectors)
                    → Database (persist indexes)

    Also handles:
        - Full vault scanning
        - Incremental change detection (SHA-256 hash)
        - Single-file reprocessing (for file watcher)
        - Deletion cleanup
    """

    def __init__(
        self,
        storage: Storage,
        chunker: Chunker,
        embedder: Embedder,
        db: Database,
    ):
        self.storage = storage
        self.chunker = chunker
        self.embedder = embedder
        self.db = db

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run_full(self) -> dict:
        """
        Scans the entire vault and indexes all changed files.
        Returns a summary dict for the Web Portal dashboard.
        """
        logger.info("Pipeline: Starting full vault scan...")
        start_time = datetime.now()

        all_files = self.storage.list_notes()
        indexed = 0
        skipped = 0
        errors = 0

        for file_path in all_files:
            relative_path = str(file_path.relative_to(self.storage.vault_path))

            try:
                if self._has_changed(relative_path, file_path):
                    self.process_file(relative_path)
                    indexed += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Pipeline: Error processing {relative_path}: {e}")
                errors += 1

        # Clean up deleted files
        deleted = self._cleanup_deleted(all_files)

        elapsed = (datetime.now() - start_time).total_seconds()
        summary = {
            "indexed": indexed,
            "skipped": skipped,
            "deleted": deleted,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(f"Pipeline: Complete. {summary}")
        return summary

    def process_file(self, relative_path: str):
        """
        Processes a single file through the full pipeline.
        Called by the file watcher on create/modify events.
        """
        logger.info(f"Pipeline: Processing {relative_path}")

        # 1. READ (Storage)
        note = self.storage.read_note(relative_path)

        # 2. CHUNK (Chunker)
        chunks = self.chunker.chunk_note(note)
        if not chunks:
            logger.warning(f"Pipeline: No chunks for {relative_path}, skipping.")
            return

        # 3. EMBED (Embedder)
        texts_to_embed = [
            f"{c.heading_path}\n\n{c.text}" for c in chunks
        ]
        embeddings = self.embedder.embed_batch(texts_to_embed)

        # 4. STORE (Database)
        self.db.upsert_note(note, chunks, embeddings)

    def remove_file(self, relative_path: str):
        """
        Removes a file's data from all indexes.
        Called by the file watcher on delete events.
        """
        logger.info(f"Pipeline: Removing {relative_path} from index")
        note_id = self.storage._generate_note_id(relative_path)
        self.db._delete_note_data(note_id)

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _has_changed(self, relative_path: str, file_path: Path) -> bool:
        """Compares file hash against stored hash to skip unchanged files."""
        current_hash = self.storage._calculate_hash(file_path)
        stored_hash = self.db.get_note_hash(relative_path)
        return current_hash != stored_hash

    def _cleanup_deleted(self, current_files: list[Path]) -> int:
        """
        Removes index entries for files that no longer exist on disk.
        Returns the number of deleted entries.
        """
        current_paths = {
            str(f.relative_to(self.storage.vault_path)) for f in current_files
        }
        stored_paths = self.db.get_all_file_paths()  # You'll add this to db.py

        deleted_paths = stored_paths - current_paths
        for path in deleted_paths:
            self.remove_file(path)

        return len(deleted_paths)