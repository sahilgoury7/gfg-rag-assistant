"""Unit Tests for API Pydantic Schemas.

Validates:
- Positive schema creation with valid bounds
- Negative validation on whitespace-only queries
- Bound constraints on top_k (1 <= top_k <= 10)
- Proper serialization of citations and health probe responses
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import Citation, HealthResponse, QueryRequest, QueryResponse


def test_query_request_valid():
    """Valid QueryRequest should instantiate without errors."""
    req = QueryRequest(query="Explain Binary Search with time complexity", top_k=5)
    assert req.query == "Explain Binary Search with time complexity"
    assert req.top_k == 5


def test_query_request_default_top_k():
    """top_k should default to 4 when omitted."""
    req = QueryRequest(query="What is Dynamic Programming?")
    assert req.top_k == 4


def test_query_request_strips_whitespace():
    """Query with leading/trailing spaces should be stripped cleanly."""
    req = QueryRequest(query="   What is Dijkstra's Algorithm?   ", top_k=3)
    assert req.query == "What is Dijkstra's Algorithm?"


def test_query_request_rejects_empty_query():
    """Empty or whitespace-only query must raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        QueryRequest(query="   ", top_k=3)
    assert "Query must contain meaningful text" in str(exc.value)


def test_query_request_top_k_bounds():
    """top_k must be strictly between 1 and 10."""
    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query", top_k=11)


def test_citation_schema():
    """Citation should properly serialize title, url, and relevance_score."""
    c = Citation(
        title="Binary Search Algorithm - GeeksforGeeks",
        url="https://www.geeksforgeeks.org/binary-search/",
        relevance_score=0.8954,
    )
    assert c.title == "Binary Search Algorithm - GeeksforGeeks"
    assert c.url == "https://www.geeksforgeeks.org/binary-search/"
    assert c.relevance_score == 0.8954


def test_health_response_schema():
    """HealthResponse schema should properly structure liveness telemetry."""
    res = HealthResponse(status="ok", version="1.0.0", environment="production")
    assert res.status == "ok"
    assert res.version == "1.0.0"
    assert res.environment == "production"
