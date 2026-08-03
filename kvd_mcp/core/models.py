from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

# --- ENUMS ---

class NoteStatus(str, Enum):
    """Standard statuses for Obsidian/Markdown notes."""
    DRAFT = "draft"
    EVERGREEN = "evergreen"
    ARCHIVED = "archived"
    SEEDLING = "seedling"

class SearchMode(str, Enum):
    """Supported search strategies."""
    HYBRID = "hybrid"     # BM25 + Vector (Default)
    SEMANTIC = "semantic" # Vector only
    KEYWORD = "keyword"   # BM25 only

# --- CORE DATA MODELS ---

class NoteMetadata(BaseModel):
    """
    Represents the YAML frontmatter and file system attributes of a Markdown note.
    Uses 'extra=allow' to preserve custom Obsidian fields (e.g., cssclass, publish) 
    without crashing the parser.
    """
    title: str
    tags: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    status: Optional[NoteStatus] = None
    
    # File system derived fields
    folder: str           # e.g., "Projects/DevOps"
    file_path: str        # Relative path: "Projects/DevOps/docker.md"
    file_hash: str        # SHA-256 of raw content (for incremental indexing)
    
    # Allow unknown frontmatter keys to pass through
    model_config = ConfigDict(extra='allow')


class Note(BaseModel):
    """
    A fully parsed Markdown note, ready for chunking.
    """
    note_id: str          # Deterministic ID (SHA-256 of file_path)
    metadata: NoteMetadata
    body: str             # Raw Markdown content (frontmatter stripped)
    
    # Populated during the indexing phase
    chunks: List["Chunk"] = Field(default_factory=list)


class Chunk(BaseModel):
    """
    A semantic slice of a Note. This is the atomic unit of search and embedding.
    """
    chunk_id: str         # Deterministic ID: "{note_id}::{index}"
    note_id: str          # Parent note ID
    index: int            # Order within the note (0, 1, 2...)
    
    # Contextual data
    heading_path: str     # e.g., "# Docker > ## Networking > ### Bridge"
    text: str             # The actual content of the chunk
    token_count: int      # Estimated token count (for context window management)


class SearchResult(BaseModel):
    """
    A single result returned from a hybrid search query.
    This is what the MCP tool returns to the AI agent.
    """
    chunk_id: str
    score: float          # RRF score or similarity distance
    text: str             # The chunk content
    heading_path: str     # Contextual location
    note_title: str       # Parent note title
    note_path: str        # Relative file path
    tags: List[str]       # Parent note tags


class IndexStats(BaseModel):
    """
    Aggregated statistics for the Web Portal dashboard.
    """
    total_notes: int
    total_chunks: int
    total_tokens: int
    last_indexed: Optional[datetime]
    vault_path: str
    db_size_mb: float
    pending_files: int    # Files detected but not yet indexed