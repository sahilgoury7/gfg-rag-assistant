"""GFG RAG Assistant — FastAPI Application Entrypoint.

This module bootstraps the full FastAPI application:
  - Lifespan context manager: warms up services on startup, cleans up on shutdown.
  - CORS middleware: allows the Next.js dashboard (or any frontend) to call the API.
  - Router registration: mounts all endpoint definitions from src/api/routes.py.
  - Uvicorn runner: allows running the server directly via `python -m src.main`.

Architecture:
  [Client] ──HTTP──> [CORS Middleware] ──> [Router] ──> [Services]
                                                    └──> [ChromaDB + LLM]
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.config import get_settings
from src.services.embedding import get_embedding_service
from src.services.generator import get_generator_service
from src.services.retriever import get_retriever_service

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: Startup & Shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle.

    ON STARTUP:
        - Warms up all singleton services (Embedding, Retriever, Generator).
        - Ensures heavy models and cloud connections are ready BEFORE
          the first user request arrives, eliminating cold-start latency.

    ON SHUTDOWN:
        - Logs graceful shutdown message.
        - Python's garbage collector handles resource cleanup automatically.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  GFG RAG Assistant — Starting Up")
    logger.info("=" * 60)

    settings = get_settings()
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"Primary LLM : {settings.primary_provider.upper()}")
    logger.info(f"Default top_k: {settings.top_k}")

    # Pre-warm services so first request has no cold start
    logger.info("Pre-warming Embedding Service (ONNX model load)...")
    get_embedding_service()

    logger.info("Pre-warming Generator Service (LLM client init)...")
    get_generator_service()

    logger.info("Pre-warming Retriever Service (ChromaDB client init)...")
    get_retriever_service()

    logger.info("All services ready. API is live.")
    logger.info("=" * 60)

    yield  # ← Application runs here, handling requests

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("GFG RAG Assistant shutting down gracefully.")


# ---------------------------------------------------------------------------
# FastAPI App Instance
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="GFG RAG Assistant API",
    description=(
        "A production-grade Retrieval-Augmented Generation (RAG) API "
        "that answers Computer Science questions grounded in 3.4 Lakh "
        "GeeksforGeeks articles indexed in ChromaDB Cloud."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI at http://localhost:8000/redoc
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------

# In development: allow all origins so you can test from browser/Postman freely.
# In production: replace "*" with your actual frontend domain (e.g. your Vercel URL).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------

# Mounts all routes defined in src/api/routes.py onto the main app.
# /health and /api/v1/query are now live.
app.include_router(router)


# ---------------------------------------------------------------------------
# Entry Point (Run directly or via Uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
        log_level="info",
    )
