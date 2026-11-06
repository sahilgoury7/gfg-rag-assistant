"""Integration and Endpoint Tests for FastAPI Application.

Validates:
- GET /health liveness probe logic & schema
- POST /api/v1/query schema validation & rejection of invalid inputs
- POST /api/v1/query full pipeline orchestration (mocked vector retrieval + LLM generation)
- Provider failover telemetry and citation score verification
- TestClient HTTP integration (enabled on Python <= 3.13 and Linux/CI environments)
"""

import sys
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError
from starlette.requests import Request

from src.api.routes import health_check, run_query
from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.services.generator import GenerationResult
from src.services.retriever import RetrievedChunk

IS_PYTHON_314_WINDOWS = sys.platform == "win32" and sys.version_info >= (3, 14)


def test_health_check_endpoint_direct():
    """health_check should return 200 HealthResponse with ok status."""
    res = health_check()
    assert isinstance(res, HealthResponse)
    assert res.status == "ok"
    assert res.version == "1.0.0"
    assert res.environment in ("development", "production")


def test_query_request_validation_rejections():
    """QueryRequest must reject empty queries and queries shorter than 3 characters."""
    with pytest.raises(ValidationError):
        QueryRequest(query="")

    with pytest.raises(ValidationError):
        QueryRequest(query="   ")

    with pytest.raises(ValidationError):
        QueryRequest(query="ab")


def test_query_request_top_k_bounds():
    """QueryRequest must enforce 1 <= top_k <= 10."""
    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query", top_k=11)


def test_run_query_e2e_pipeline():
    """run_query should orchestrate retrieval, context formatting, LLM generation, and return QueryResponse."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/query",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }
    request = Request(scope)
    payload = QueryRequest(query="What is binary search and its time complexity?", top_k=3)

    mock_chunk = RetrievedChunk(
        chunk_id="chunk_101",
        text="Binary search divides the array in half each step with O(log n) complexity.",
        source_url="https://geeksforgeeks.org/binary-search",
        title="Binary Search Algorithm",
        distance=0.15,
    )
    mock_gen = GenerationResult(
        answer="Binary search runs in O(log n) time by dividing the sorted search space.",
        model_used="openai/gpt-oss-120b",
        provider="groq",
        fallback_triggered=False,
    )

    with patch("src.api.routes.get_retriever_service") as mock_retriever_fn, \
         patch("src.api.routes.get_generator_service") as mock_generator_fn:

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_chunk]
        mock_retriever.format_context.return_value = "Formatted GFG Context"
        mock_retriever_fn.return_value = mock_retriever

        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_gen
        mock_generator_fn.return_value = mock_generator

        response = run_query(request, payload)

        assert isinstance(response, QueryResponse)
        assert response.query == payload.query
        assert "Binary search runs in O(log n)" in response.answer
        assert len(response.citations) == 1
        assert response.citations[0].title == "Binary Search Algorithm"
        assert response.citations[0].url == "https://geeksforgeeks.org/binary-search"
        assert round(response.citations[0].relevance_score, 2) == 0.85
        assert response.provider_used == "groq"
        assert response.fallback_triggered is False
        assert response.latency_ms > 0


def test_run_query_fallback_telemetry():
    """run_query should correctly report provider_used='google' and fallback_triggered=True."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/query",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }
    request = Request(scope)
    payload = QueryRequest(query="Explain Dynamic Programming", top_k=2)

    mock_gen_fallback = GenerationResult(
        answer="Dynamic programming solves problems by breaking them into overlapping subproblems.",
        model_used="gemini-2.5-flash",
        provider="google",
        fallback_triggered=True,
    )

    with patch("src.api.routes.get_retriever_service") as mock_retriever_fn, \
         patch("src.api.routes.get_generator_service") as mock_generator_fn:

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever.format_context.return_value = "No relevant context found."
        mock_retriever_fn.return_value = mock_retriever

        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_gen_fallback
        mock_generator_fn.return_value = mock_generator

        response = run_query(request, payload)

        assert response.provider_used == "google"
        assert response.fallback_triggered is True
        assert len(response.citations) == 0


def test_rate_limiter_proxy_ip_resolution():
    """get_real_client_ip must extract genuine client IP from X-Forwarded-For header."""
    from src.core.limiter import get_real_client_ip

    # Case 1: Standard direct request without proxy headers
    scope_direct = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/query",
        "headers": [],
        "client": ("192.168.1.50", 54321),
    }
    req_direct = Request(scope_direct)
    assert get_real_client_ip(req_direct) == "192.168.1.50"

    # Case 2: Behind Cloud reverse proxy (Render / Cloudflare / ALB)
    scope_proxy = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/query",
        "headers": [(b"x-forwarded-for", b"203.0.113.195, 10.0.0.1")],
        "client": ("10.0.0.1", 80),  # Internal proxy IP
    }
    req_proxy = Request(scope_proxy)
    assert get_real_client_ip(req_proxy) == "203.0.113.195"


@pytest.mark.skipif(
    IS_PYTHON_314_WINDOWS,
    reason="AnyIO blocking portal has known deadlock on Windows CPython 3.14 (CI runs on Python 3.12)",
)
def test_testclient_http_integration():
    """HTTP TestClient integration test for Python 3.12 / Linux CI."""
    from fastapi.testclient import TestClient
    from src.main import app

    with patch("src.main.get_embedding_service"), \
         patch("src.main.get_generator_service"), \
         patch("src.main.get_retriever_service"):
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"
