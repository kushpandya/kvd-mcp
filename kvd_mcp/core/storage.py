# src/kvd_mcp/core/storage.py
import frontmatter
from pathlib import Path
from typing import List, Optional
import logging

from .models import Note, NoteMetadata

logger = logging.getLogger(__name__)

class Storage:
    """ Lowest layer: Handles all direct File I/O for the Markdown vault.
    Guarantees safe path resolution and consistent frontmatter parsing.
    """

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path.resolve()
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault_path}")

    def list_notes(self) -> List[Path]:
        """Recursively finds all .md files in the vault."""
        return list(self.vault_path.rglob("*.md"))

    def read_note(self, relative_path: str) -> Note:
        """
        Reads a specific note from disk, parses frontmatter, and returns a Note object.
        """
        full_path = self._resolve_safe_path(relative_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Note not found: {relative_path}")

        # Parse frontmatter and body
        post = frontmatter.load(full_path)
        
        # Extract metadata (using .get to prevent crashes on missing keys)
        metadata = NoteMetadata(
            title=post.get("title", full_path.stem),
            tags=post.get("tags", []),
            aliases=post.get("aliases", []),
            created=post.get("created"),
            modified=post.get("modified"),
            status=post.get("status"),
            folder=str(full_path.parent.relative_to(self.vault_path)),
            file_path=relative_path,
            file_hash=self._calculate_hash(full_path),
            extra={k: v for k, v in post.metadata.items() if k not in NoteMetadata.model_fields}
        )

        return Note(
            note_id=self._generate_note_id(relative_path),
            metadata=metadata,
            body=post.content
        )

    def write_note(self, relative_path: str, metadata: NoteMetadata, body: str):
        """
        Creates or updates a note on disk. Safely serializes frontmatter.
        """
        full_path = self._resolve_safe_path(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Build frontmatter dictionary
        fm_dict = {
            "title": metadata.title,
            "tags": metadata.tags,
            "aliases": metadata.aliases,
            "created": metadata.created.isoformat() if metadata.created else None,
            "modified": metadata.modified.isoformat() if metadata.modified else None,
            "status": metadata.status.value if metadata.status else None,
        }
        # Merge in any extra fields
        fm_dict.update(metadata.extra)
        
        # Remove None values to keep YAML clean
        fm_dict = {k: v for k, v in fm_dict.items() if v is not None}

        post = frontmatter.Post(body, **fm_dict)
        
        # Write to disk
        with open(full_path, "w", encoding="utf-8") as f:
            frontmatter.dump(post, f)
            
        logger.info(f"Successfully wrote note: {relative_path}")

    def delete_note(self, relative_path: str):
        """Safely deletes a note from the vault."""
        full_path = self._resolve_safe_path(relative_path)
        if full_path.exists():
            full_path.unlink()
            logger.info(f"Deleted note: {relative_path}")

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """
        SECURITY CRITICAL: Prevents directory traversal attacks (e.g., ../../etc/passwd).
        Ensures the resolved path is strictly inside the vault directory.
        """
        target = (self.vault_path / relative_path).resolve()
        if not str(target).startswith(str(self.vault_path)):
            raise PermissionError(f"Access denied: Path escapes vault directory ({relative_path})")
        return target

    def _calculate_hash(self, file_path: Path) -> str:
        """Generates a SHA-256 hash of the file content for change detection."""
        import hashlib
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def _generate_note_id(self, relative_path: str) -> str:
        """Deterministic ID based on file path."""
        import hashlib
        return hashlib.sha256(relative_path.encode()).hexdigest()[:16]