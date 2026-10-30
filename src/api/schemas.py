"""API Schemas Module.

Defines the strict Pydantic v2 contracts (data shapes) for all
incoming HTTP requests and outgoing HTTP responses.

Think of these as the "official menu" of the RAG API:
- What the client MUST send (QueryRequest)
- What the server WILL return (QueryResponse)
Any deviation is rejected at the gateway before touching any service.
"""

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# REQUEST MODELS (Client → Server)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Contract for incoming RAG query requests.

    Enforces strict validation so garbage input never reaches
    the expensive Chroma / LLM pipeline.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's natural language question about a CS/programming topic.",
        examples=["What is dynamic programming?"],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of GFG chunks to retrieve. Must be between 1 and 10.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        """Rejects queries that are only whitespace."""
        if not v.strip():
            raise ValueError("Query must contain meaningful text, not just whitespace.")
        return v.strip()


# ---------------------------------------------------------------------------
# RESPONSE MODELS (Server → Client)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """Represents a single cited GFG source article for an answer."""

    title: str = Field(description="Title of the GeeksforGeeks article.")
    url: str | None = Field(
        default=None,
        description="Direct URL to the source GFG article (if available).",
    )
    relevance_score: float = Field(
        description="Semantic similarity score between 0.0 (irrelevant) and 1.0 (perfect match).",
        ge=0.0,
        le=1.0,
    )


class QueryResponse(BaseModel):
    """Full structured response returned after a successful RAG pipeline execution."""

    query: str = Field(description="The original query echoed back for confirmation.")
    answer: str = Field(description="LLM-generated answer grounded in GFG context.")
    citations: list[Citation] = Field(
        description="Ranked list of GFG source chunks used to generate the answer."
    )
    provider_used: str = Field(
        description="LLM provider that generated the answer (e.g. 'groq', 'google')."
    )
    fallback_triggered: bool = Field(
        description="True if the primary LLM failed and the fallback provider was used."
    )
    latency_ms: float = Field(
        description="End-to-end request processing time in milliseconds."
    )


class HealthResponse(BaseModel):
    """Lightweight health-check response for uptime monitoring."""

    status: str = Field(description="'ok' when all services are operational.")
    version: str = Field(description="Current application version string.")
    environment: str = Field(description="Runtime environment: development / production.")
