"""Unit Tests for Vector Retriever Service.

Validates:
- Cosine distance to relevance score conversion
- Document context formatting into LLM-ready prompts
- Handling of empty chunk lists
"""

from unittest.mock import MagicMock
from src.services.retriever import RetrievedChunk, RetrieverService


def test_chunk_relevance_score_conversion():
    """Distance should convert to relevance score as max(0.0, 1.0 - distance)."""
    chunk_high = RetrievedChunk(
        chunk_id="chunk_1",
        text="Binary Search divides the search space in half.",
        source_url="https://geeksforgeeks.org/binary-search",
        title="Binary Search",
        distance=0.15,
    )
    assert round(chunk_high.relevance_score, 2) == 0.85

    chunk_far = RetrievedChunk(
        chunk_id="chunk_2",
        text="Unrelated content.",
        source_url=None,
        title="Unrelated",
        distance=1.25,
    )
    assert chunk_far.relevance_score == 0.0


def test_format_context_multiple_chunks():
    """format_context should format chunks with document markers and titles."""
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Python was created by Guido van Rossum.",
            source_url="https://geeksforgeeks.org/python",
            title="Python History",
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Dynamic typing allows variables to change types.",
            source_url="https://geeksforgeeks.org/dynamic-typing",
            title="Dynamic Typing",
            distance=0.2,
        ),
    ]

    context = RetrieverService.format_context(chunks)
    assert "--- Document [1] [Source: Python History] ---" in context
    assert "Python was created by Guido van Rossum." in context
    assert "--- Document [2] [Source: Dynamic Typing] ---" in context
    assert "Dynamic typing allows variables to change types." in context


def test_format_context_empty():
    """Empty chunk list should return a clean fallback string."""
    assert RetrieverService.format_context([]) == "No relevant context found."


def test_retriever_query_parsing():
    """Retriever should correctly map ChromaDB query results into RetrievedChunk objects."""
    mock_settings = MagicMock()
    mock_settings.top_k = 2
    mock_settings.chroma_collection = "articles"

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 384

    retriever = RetrieverService(settings=mock_settings, embedding_service=mock_embedding)

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk_101"]],
        "documents": [["Merge sort has O(n log n) time complexity."]],
        "metadatas": [[{"title": "Merge Sort", "link": "https://geeksforgeeks.org/merge-sort"}]],
        "distances": [[0.22]],
    }
    retriever._collection = mock_collection

    results = retriever.retrieve("What is merge sort?", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_101"
    assert results[0].title == "Merge Sort"
    assert results[0].source_url == "https://geeksforgeeks.org/merge-sort"
    assert round(results[0].distance, 2) == 0.22
    assert round(results[0].relevance_score, 2) == 0.78

