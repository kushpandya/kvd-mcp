# kvd_mcp/core/embeddings.py
import ollama
from typing import List
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.dimension = self._get_dimension()  # ← AUTO-DETECT
        self._verify_connection()

    def _get_dimension(self) -> int:
        """Queries Ollama to get the embedding dimension for this model."""
        try:
            # Generate a test embedding to detect dimension
            response = ollama.embed(model=self.model_name, input="test")
            return len(response["embeddings"][0])
        except Exception as e:
            logger.error(f"Failed to detect embedding dimension: {e}")
            # Fallback defaults
            if "nomic" in self.model_name:
                return 768
            elif "minilm" in self.model_name:
                return 384
            else:
                return 768  # Default to 768

    def _verify_connection(self):
        try:
            ollama.list()
            logger.info(f"Embedder initialized: {self.model_name} ({self.dimension} dims)")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            return []
        response = ollama.embed(model=self.model_name, input=text)
        return response["embeddings"][0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        valid_texts = [t if t.strip() else "empty" for t in texts]
        response = ollama.embed(model=self.model_name, input=valid_texts)
        return response["embeddings"]