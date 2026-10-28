"""Retriever Service Module.

Connects to ChromaDB Cloud (housing 3.4 Lakh indexed GFG chunks),
queries nearest neighbors using ONNX embeddings, and extracts
grounded text chunks along with source metadata (URLs, titles).
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any
import chromadb
from src.core.config import Settings, get_settings
from src.services.embedding import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a single retrieved document chunk with source metadata."""

    chunk_id: str
    text: str
    source_url: str | None
    title: str | None
    distance: float

    @property
    def relevance_score(self) -> float:
        """Converts cosine distance into a similarity score (0.0 to 1.0)."""
        return max(0.0, 1.0 - self.distance)


class RetrieverService:
    """Handles semantic search over Chroma Cloud collection."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or get_embedding_service()
        self._collection = None

    def _get_collection(self) -> Any:
        """Lazily connects to Chroma Cloud client and returns the collection.
        
        Lazy initialization avoids blocking server boot if network is slow.
        """
        if self._collection is None:
            logger.info(
                f"Connecting to Chroma Cloud (tenant: {self._settings.chroma_tenant}, "
                f"db: {self._settings.chroma_database})..."
            )
            client = chromadb.CloudClient(
                api_key=self._settings.chroma_api_key,
                tenant=self._settings.chroma_tenant,
                database=self._settings.chroma_database,
            )
            self._collection = client.get_collection(
                name=self._settings.chroma_collection
            )
            logger.info(
                f"Successfully connected to collection: '{self._settings.chroma_collection}'"
            )
        return self._collection

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Performs vector search in Chroma Cloud and returns top-K relevant chunks."""
        k = top_k or self._settings.top_k
        query_vector = self._embedding_service.embed_query(query)

        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for i in range(len(documents)):
            doc_text = documents[i]
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            dist = distances[i] if i < len(distances) else 0.0
            chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"

            # Extract URL and Title gracefully across different possible metadata keys
            source_url = (
                meta.get("url")
                or meta.get("link")
                or meta.get("source")
                or None
            )
            title = meta.get("title") or meta.get("heading") or "GeeksforGeeks Article"

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc_text,
                    source_url=source_url,
                    title=title,
                    distance=float(dist),
                )
            )

        return chunks

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Formats a list of retrieved chunks into an LLM-ready context block."""
        if not chunks:
            return "No relevant context found."

        formatted_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source_tag = f" [Source: {chunk.title}]" if chunk.title else ""
            formatted_blocks.append(
                f"--- Document [{idx}]{source_tag} ---\n{chunk.text}"
            )

        return "\n\n".join(formatted_blocks)


@lru_cache()
def get_retriever_service() -> RetrieverService:
    """Returns a cached singleton instance of RetrieverService."""
    return RetrieverService()
