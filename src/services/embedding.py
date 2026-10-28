"""Embedding Service Module.

Provides a memory-efficient, ONNX-backed embedding wrapper using Chroma's
DefaultEmbeddingFunction (all-MiniLM-L6-v2, 384 dimensions).
Strictly operates within sub-120MB RAM footprint without heavy PyTorch.
"""

from functools import lru_cache
import logging
from typing import Sequence
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper around Chroma's native ONNX embedding function."""

    def __init__(self) -> None:
        """Initializes the lightweight ONNX DefaultEmbeddingFunction."""
        logger.info("Initializing ONNX DefaultEmbeddingFunction (all-MiniLM-L6-v2)...")
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self.embedding_dimension = 384

    def embed_query(self, query: str) -> list[float]:
        """Embeds a single query string into a 384-dim normalized vector."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")

        embeddings = self._ef([query.strip()])
        return embeddings[0]

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Embeds a batch of document strings into 384-dim vectors."""
        if not documents:
            return []
        return self._ef(list(documents))


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Returns a singleton instance of the EmbeddingService.
    
    Prevents redundant model re-instantiation and keeps memory strictly bounded.
    """
    return EmbeddingService()
