# kvd_mcp/core/search.py
import re
from typing import List, Optional
from dataclasses import dataclass

from .models import SearchResult, SearchMode
from .storage import Storage
from .embeddings import Embedder
from .db import Database

@dataclass
class SearchConfig:
    limit: int = 10
    rrf_k: int = 60
    max_context_tokens: int = 4000

class SearchEngine:
    """
    The intelligent retrieval engine.
    Orchestrates hybrid search and formats context for LLMs.
    """

    def __init__(self, storage: Storage, embedder: Embedder, db: Database):
        self.storage = storage
        self.embedder = embedder
        self.db = db

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def search(
        self, 
        query: str, 
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 10  # ← Simple, direct limit argument
    ) -> List[SearchResult]:
        """
        Main entry point for searching the vault.
        """
        if not query.strip():
            return []

        clean_query = self._sanitize_query(query)
        config = SearchConfig(limit=limit) # ← Wrap limit in config for private methods

        if mode == SearchMode.KEYWORD:
            return self._lexical_search(clean_query, config)
        elif mode == SearchMode.SEMANTIC:
            return self._semantic_search(clean_query, config)
        else:
            return self._hybrid_search(clean_query, config)

    def format_context(self, results: List[SearchResult], max_tokens: int = 4000) -> str:
        """
        Packages search results into a clean, structured text block for the AI agent.
        """
        if not results:
            return "No relevant context found in the vault."

        context_parts = []
        current_tokens = 0

        for res in results:
            chunk_block = (
                f"---\n"
                f"Source: {res.note_title} ({res.note_path})\n"
                f"Section: {res.heading_path}\n"
                f"Tags: {', '.join(res.tags) if res.tags else 'None'}\n\n"
                f"{res.text}\n"
            )
            
            est_tokens = len(chunk_block) // 4
            
            if current_tokens + est_tokens > max_tokens:
                break
                
            context_parts.append(chunk_block)
            current_tokens += est_tokens

        return "\n".join(context_parts)

    # =========================================================================
    # PRIVATE SEARCH STRATEGIES
    # =========================================================================

    def _lexical_search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Pure BM25 keyword search via FTS5."""
        raw_chunk_ids = self.db.query_fts5(query, limit=config.limit * 2)
        return self._enrich_results(raw_chunk_ids)

    def _semantic_search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Pure vector cosine similarity search via sqlite-vec."""
        query_vec = self.embedder.embed_text(query)
        raw_chunk_ids = self.db.query_vectors(query_vec, limit=config.limit * 2)
        return self._enrich_results(raw_chunk_ids)

    def _hybrid_search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """
        Combines Lexical and Semantic search using Reciprocal Rank Fusion (RRF).
        """
        # 1. Get Lexical Rankings
        lexical_ids = self.db.query_fts5(query, limit=config.limit * 2)
        
        # 2. Get Semantic Rankings
        query_vec = self.embedder.embed_text(query)
        semantic_ids = self.db.query_vectors(query_vec, limit=config.limit * 2)

        # 3. Apply RRF Math
        scores = {}
        
        for rank, chunk_id in enumerate(lexical_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (config.rrf_k + rank))
            
        for rank, chunk_id in enumerate(semantic_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (config.rrf_k + rank))

        # 4. Sort by combined RRF score
        top_chunk_ids = sorted(scores, key=scores.get, reverse=True)[:config.limit]
        
        # 5. Fetch full details for the winners and attach scores
        return self._enrich_results(top_chunk_ids, scores)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _sanitize_query(self, query: str) -> str:
        """Removes FTS5 special characters to prevent database crashes."""
        return re.sub(r'[*+\-()"]', ' ', query).strip()

    def _enrich_results(self, chunk_ids: List[str], scores: dict = None) -> List[SearchResult]:
        """Takes raw chunk IDs from the DB and fetches full metadata."""
        if not chunk_ids:
            return []
            
        raw_data = self.db.get_chunks_by_ids(chunk_ids)
        results = []
        
        for row in raw_data:
            chunk_id = row['chunk_id']
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=scores.get(chunk_id, 0.0) if scores else 0.0,
                text=row['text'],
                heading_path=row['heading_path'],
                note_title=row['note_title'],
                note_path=row['note_path'],
                tags=row['tags']
            ))
            
        # If we passed scores, sort by them to maintain RRF ranking
        if scores:
            results.sort(key=lambda x: x.score, reverse=True)
            
        return results