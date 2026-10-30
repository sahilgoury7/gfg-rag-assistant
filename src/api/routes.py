"""API Routes Module.

Defines all HTTP endpoints for the GFG RAG Assistant API:
  - GET  /health            → Liveness probe for cloud monitoring
  - POST /api/v1/query      → Main RAG pipeline endpoint

Route logic follows the Controller pattern:
  1. Accept validated request (Pydantic guarantees clean data)
  2. Orchestrate services (Embed → Retrieve → Generate)
  3. Return structured response (predictable JSON contract)
"""

import logging
import time

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import Citation, HealthResponse, QueryRequest, QueryResponse
from src.core.config import get_settings
from src.services.generator import get_generator_service
from src.services.retriever import get_retriever_service

logger = logging.getLogger(__name__)

# Single router instance — mounted by main.py onto the FastAPI app
router = APIRouter()


# ---------------------------------------------------------------------------
# HEALTH CHECK ENDPOINT
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description=(
        "Returns 200 OK when the API server is running. "
        "Used by cloud platforms (Render, Docker, AWS ECS) to verify the pod is alive."
    ),
    tags=["Monitoring"],
)
def health_check() -> HealthResponse:
    """Lightweight liveness probe — no heavy service calls."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment=settings.app_env,
    )


# ---------------------------------------------------------------------------
# MAIN RAG QUERY ENDPOINT
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Run RAG Query",
    description=(
        "Accepts a natural language question, retrieves the most relevant GFG article "
        "chunks from ChromaDB, and generates a grounded answer via the Dual-LLM engine "
        "(Groq primary → Gemini fallback). Returns the answer, citations, and telemetry."
    ),
    tags=["RAG Pipeline"],
)
def run_query(request: QueryRequest) -> QueryResponse:
    """Orchestrates the full Embed → Retrieve → Generate pipeline.

    Args:
        request: Validated QueryRequest containing the user query and top_k setting.

    Returns:
        QueryResponse with answer, GFG citations, provider telemetry, and latency.

    Raises:
        HTTPException 503: When both primary and fallback LLM providers fail.
        HTTPException 500: For any unexpected internal server error.
    """
    # ── Start latency timer ──────────────────────────────────────────────────
    start_time = time.perf_counter()

    logger.info(f"Incoming query: '{request.query}' | top_k={request.top_k}")

    try:
        # ── Step 1: Retrieve relevant GFG chunks from ChromaDB ───────────────
        retriever = get_retriever_service()
        chunks = retriever.retrieve(query=request.query, top_k=request.top_k)

        if not chunks:
            logger.warning("ChromaDB returned zero chunks — collection may be empty.")

        # ── Step 2: Format chunks into LLM-ready context block ───────────────
        context = retriever.format_context(chunks)

        # ── Step 3: Generate answer via Dual-LLM Circuit Breaker ─────────────
        generator = get_generator_service()
        result = generator.generate(query=request.query, context=context)

        # ── Step 4: Build citation list from retrieved chunks ─────────────────
        citations = [
            Citation(
                title=chunk.title or "GeeksforGeeks Article",
                url=chunk.source_url,
                relevance_score=round(chunk.relevance_score, 4),
            )
            for chunk in chunks
        ]

        # ── Step 5: Calculate end-to-end latency ─────────────────────────────
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            f"Query completed | provider={result.provider} | "
            f"fallback={result.fallback_triggered} | latency={latency_ms}ms"
        )

        return QueryResponse(
            query=request.query,
            answer=result.answer,
            citations=citations,
            provider_used=result.provider,
            fallback_triggered=result.fallback_triggered,
            latency_ms=latency_ms,
        )

    except RuntimeError as exc:
        # Both LLM providers failed — Circuit Breaker exhausted
        logger.error(f"All LLM providers exhausted: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The answer generation service is temporarily unavailable. "
                "Both primary (Groq) and fallback (Gemini) providers failed. "
                "Please retry in a few moments."
            ),
        ) from exc

    except Exception as exc:
        # Unexpected error — log full traceback for debugging
        logger.exception(f"Unexpected error during query processing: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred. Please check server logs.",
        ) from exc
