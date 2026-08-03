from pathlib import Path
from kvd_mcp.core.storage import Storage
from kvd_mcp.core.chunker import Chunker, ChunkerConfig
from kvd_mcp.core.embeddings import Embedder
from kvd_mcp.core.db import Database
from kvd_mcp.core.pipeline import Pipeline

def main():
    VAULT_DIR = Path("./test_vault")
    DB_FILE = Path("./test.db")

    VAULT_DIR.mkdir(exist_ok=True)
    (VAULT_DIR / "test.md").write_text(
        "---\n"
        "title: Test Note\n"
        "tags: [test]\n"
        "---\n\n"
        "# Hello World\n\n"
        "This is a test of the indexing pipeline."
    )

    storage = Storage(VAULT_DIR)
    chunker = Chunker(ChunkerConfig(max_chunk_tokens=400))
    embedder = Embedder(model_name="nomic-embed-text")
    
    # CRITICAL: Pass embedder.dimension to Database
    db = Database(DB_FILE, embedding_dim=embedder.dimension)

    pipeline = Pipeline(storage, chunker, embedder, db)

    summary = pipeline.run_full()
    print("\n✅ Pipeline Summary:")
    print(summary)

    from kvd_mcp.core.search import SearchEngine, SearchConfig
    search_engine = SearchEngine(storage, embedder, db)
    results = search_engine.search("Naam")
    print(f"\n✅ Search found {len(results)} results:")
    for r in results:
        print(f"  - {r.note_title}: {r.text[:80]}...")

if __name__ == "__main__":
    main()
