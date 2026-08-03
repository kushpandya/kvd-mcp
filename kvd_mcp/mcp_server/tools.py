# kvd_mcp/mcp_server/tools.py
"""
MCP Tool Definitions for kvd-mcp.

These tools are exposed to AI agents via the Model Context Protocol.
Each function becomes a callable tool with the docstring as its description.
"""
from typing import Optional
from datetime import datetime

from ..app import get_app
from ..core.models import SearchMode, NoteMetadata


# ============================================================================
# SEARCH & RETRIEVAL TOOLS
# ============================================================================

def search(query: str, limit: int = 10, mode: str = "hybrid") -> str:
    """
    Search the vault using hybrid search (keyword + semantic).
    
    Args:
        query: The search query (natural language or keywords)
        limit: Maximum number of results to return (default: 10)
        mode: Search mode - "hybrid" (default), "keyword", or "semantic"
    
    Returns:
        Formatted context with relevant chunks, including source attribution
    """
    app = get_app()
    
    # Map string mode to enum
    mode_map = {
        "hybrid": SearchMode.HYBRID,
        "keyword": SearchMode.KEYWORD,
        "semantic": SearchMode.SEMANTIC,
    }
    search_mode = mode_map.get(mode.lower(), SearchMode.HYBRID)
    
    results = app.search.search(query, mode=search_mode, limit=limit)
    
    if not results:
        return f"No results found for query: '{query}'"
    
    return app.search.format_context(results, max_tokens=4000)


def get_context(query: str, max_tokens: int = 4000) -> str:
    """
    Get smart context packing for a query.
    Optimized for LLM context windows with source attribution.
    
    Args:
        query: The query to find context for
        max_tokens: Maximum tokens to return (default: 4000)
    
    Returns:
        Structured context blocks ready for LLM consumption
    """
    app = get_app()
    results = app.search.search(query, limit=20)  # Get more candidates
    return app.search.format_context(results, max_tokens=max_tokens)


def read_note(file_path: str) -> str:
    """
    Read the full content of a specific note by its relative path.
    
    Args:
        file_path: Relative path to the note (e.g., "Projects/docker.md")
    
    Returns:
        The full Markdown content of the note
    """
    app = get_app()
    try:
        note = app.storage.read_note(file_path)
        return f"# {note.metadata.title}\n\n{note.body}"
    except FileNotFoundError:
        return f"Error: Note not found at '{file_path}'"
    except Exception as e:
        return f"Error reading note: {str(e)}"


def list_notes(folder: Optional[str] = None, tag: Optional[str] = None, limit: int = 50) -> str:
    """
    List notes in the vault, optionally filtered by folder or tag.
    
    Args:
        folder: Filter by folder path (e.g., "Projects/DevOps")
        tag: Filter by tag (e.g., "docker")
        limit: Maximum number of notes to return (default: 50)
    
    Returns:
        A list of note titles and paths
    """
    app = get_app()
    
    # Query the database for notes
    query = "SELECT file_path, title, tags FROM notes WHERE 1=1"
    params = []
    
    if folder:
        query += " AND folder LIKE ?"
        params.append(f"%{folder}%")
    
    if tag:
        query += " AND tags LIKE ?"
        params.append(f'%"{tag}"%')  # JSON array contains tag
    
    query += " ORDER BY indexed_at DESC LIMIT ?"
    params.append(limit)
    
    rows = app.db.conn.execute(query, params).fetchall()
    
    if not rows:
        return "No notes found matching the criteria."
    
    output = []
    for row in rows:
        file_path, title, tags_json = row
        output.append(f"- {title or file_path} ({file_path})")
    
    return "\n".join(output)


# ============================================================================
# NOTE CREATION & EDITING TOOLS
# ============================================================================

def create_note(
    file_path: str,
    title: str,
    content: str,
    tags: Optional[list[str]] = None,
    status: Optional[str] = None
) -> str:
    """
    Create a new note in the vault.
    
    Args:
        file_path: Relative path for the new note (e.g., "Projects/new-note.md")
        title: The note title (will be added to frontmatter)
        content: The Markdown content (without frontmatter)
        tags: Optional list of tags
        status: Optional status (e.g., "draft", "evergreen")
    
    Returns:
        Success message with the created file path
    """
    app = get_app()
    
    try:
        metadata = NoteMetadata(
            title=title,
            tags=tags or [],
            aliases=[],
            folder="/".join(file_path.split("/")[:-1]) or "",
            file_path=file_path,
            file_hash="",  # Will be calculated by storage
            status=status,
        )
        
        app.storage.write_note(file_path, metadata, content)
        
        # Trigger indexing of the new file
        app.pipeline.process_file(file_path)
        
        return f"✅ Created note: {file_path}"
    except Exception as e:
        return f"❌ Error creating note: {str(e)}"


def update_note(file_path: str, content: str) -> str:
    """
    Update the content of an existing note.
    
    Args:
        file_path: Relative path to the note to update
        content: The new Markdown content (frontmatter will be preserved)
    
    Returns:
        Success message
    """
    app = get_app()
    
    try:
        # Read existing note to preserve frontmatter
        existing_note = app.storage.read_note(file_path)
        
        # Write updated content with same metadata
        app.storage.write_note(file_path, existing_note.metadata, content)
        
        # Re-index the updated file
        app.pipeline.process_file(file_path)
        
        return f"✅ Updated note: {file_path}"
    except FileNotFoundError:
        return f"❌ Error: Note not found at '{file_path}'"
    except Exception as e:
        return f"❌ Error updating note: {str(e)}"


def delete_note(file_path: str) -> str:
    """
    Delete a note from the vault and remove it from the index.
    
    Args:
        file_path: Relative path to the note to delete
    
    Returns:
        Success message
    """
    app = get_app()
    
    try:
        app.storage.delete_note(file_path)
        app.pipeline.remove_file(file_path)
        return f"✅ Deleted note: {file_path}"
    except Exception as e:
        return f"❌ Error deleting note: {str(e)}"


# ============================================================================
# INDEXING & MAINTENANCE TOOLS
# ============================================================================

def reindex_vault() -> str:
    """
    Trigger a full re-index of the vault.
    Useful after bulk file changes or if the index seems out of sync.
    
    Returns:
        Summary of the indexing operation
    """
    app = get_app()
    summary = app.pipeline.run_full()
    
    return (
        f"✅ Indexing complete\n"
        f"- Indexed: {summary['indexed']}\n"
        f"- Skipped (unchanged): {summary['skipped']}\n"
        f"- Deleted: {summary['deleted']}\n"
        f"- Errors: {summary['errors']}\n"
        f"- Time: {summary['elapsed_seconds']}s"
    )


def get_index_stats() -> str:
    """
    Get statistics about the current index.
    
    Returns:
        Stats including total notes, chunks, and database size
    """
    app = get_app()
    
    # Query database for stats
    total_notes = app.db.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    total_chunks = app.db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    
    db_size_mb = app.db.db_path.stat().st_size / (1024 * 1024)
    
    return (
        f"📊 Index Statistics\n"
        f"- Total Notes: {total_notes}\n"
        f"- Total Chunks: {total_chunks}\n"
        f"- Database Size: {db_size_mb:.2f} MB\n"
        f"- Vault Path: {app.storage.vault_path}"
    )


def health_check() -> str:
    """
    Check if the MCP server and all components are healthy.
    
    Returns:
        Health status message
    """
    try:
        app = get_app()
        
        # Check if storage is accessible
        vault_exists = app.storage.vault_path.exists()
        
        # Check if database is accessible
        db_exists = app.db.db_path.exists()
        
        # Check if embedder is working
        test_embedding = app.embedder.embed_text("test")
        embedder_working = len(test_embedding) > 0
        
        if vault_exists and db_exists and embedder_working:
            return "✅ All systems operational"
        else:
            issues = []
            if not vault_exists:
                issues.append("Vault directory not found")
            if not db_exists:
                issues.append("Database file not found")
            if not embedder_working:
                issues.append("Embedder not responding")
            return f"⚠️ Issues detected: {', '.join(issues)}"
    
    except Exception as e:
        return f"❌ Health check failed: {str(e)}"