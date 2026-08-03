# src/kvd_mcp/core/config.py
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Centralized configuration. 
    Loads from environment variables or .env file.
    """
    # Paths
    vault_path: Path = Field(default=Path("./data/vault"))
    db_path: Path = Field(default=Path("./data/kvd.db"))
    
    # Embedding model
    embedding_model: str = Field(default="nomic-embed-text")
    embedding_dimensions: int = Field(default=768)  # Match your model!
    
    # Chunking
    max_chunk_tokens: int = Field(default=400)
    min_chunk_tokens: int = Field(default=80)
    overlap_tokens: int = Field(default=50)
    
    # Search
    default_search_limit: int = Field(default=10)
    rrf_k: int = Field(default=60)
    max_context_tokens: int = Field(default=4000)
    
    # Server
    mcp_port: int = Field(default=8080)
    web_port: int = Field(default=8000)
    
    model_config = {
        "env_prefix": "KVD_",  # Reads KVD_VAULT_PATH, KVD_DB_PATH, etc.
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

# Singleton instance
settings = Settings()