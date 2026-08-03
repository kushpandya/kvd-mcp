# kvd_mcp/app.py
"""
Application factory for kvd-mcp.
Wires all components together. Both MCP and Web Portal use this.
"""
import logging
from pathlib import Path

from .core.config import settings
from .core.storage import Storage
from .core.chunker import Chunker, ChunkerConfig
from .core.embeddings import Embedder
from .core.db import Database
from .core.pipeline import Pipeline  # ← UPDATED IMPORT
from .core.search import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KvdApp:
    """The application container holding all initialized components."""
    
    def __init__(self):
        logger.info(f"Initializing kvd-mcp with vault: {settings.vault_path}")
        
        settings.vault_path.mkdir(parents=True, exist_ok=True)
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize base components
        self.storage = Storage(settings.vault_path)
        self.embedder = Embedder(settings.embedding_model)
        self.db = Database(settings.db_path)
        
        self.chunker = Chunker(ChunkerConfig(
            max_chunk_tokens=settings.max_chunk_tokens,
            min_chunk_tokens=settings.min_chunk_tokens,
            overlap_tokens=settings.overlap_tokens,
        ))
        
        # 2. Wire the Pipeline (Write Orchestrator)
        self.pipeline = Pipeline(  # ← UPDATED NAME
            storage=self.storage,
            chunker=self.chunker,
            embedder=self.embedder,
            db=self.db,
        )
        
        # 3. Wire the Search Engine (Read Orchestrator)
        self.search = SearchEngine(
            storage=self.storage,
            embedder=self.embedder,
            db=self.db,
        )
        
        logger.info("kvd-mcp initialized successfully")
    
    def run_initial_index(self):
        """Run a full index on startup. Returns the summary dict."""
        logger.info("Running initial index...")
        summary = self.pipeline.run_full()  # ← UPDATED METHOD CALL
        logger.info(f"Index complete: {summary}")
        return summary


# Global singleton
_app: KvdApp | None = None

def get_app() -> KvdApp:
    """Get or create the global application instance."""
    global _app
    if _app is None:
        _app = KvdApp()
    return _app