"""Central Application Configuration Module.

Uses Pydantic v2 BaseSettings to validate, parse, and enforce all
environment variables and application constants from .env.
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration schema for GFG RAG Assistant."""

    # Application Environment
    app_env: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Current runtime environment"
    )

    # ChromaDB Cloud Configuration (3.4 Lakh Indexed GFG Chunks)
    chroma_api_key: str = Field(
        ...,
        description="Chroma Cloud API Token for authentication"
    )
    chroma_tenant: str = Field(
        default="f98bf1bd-b7ed-4d64-90a8-9c4be572e360",
        description="Chroma Cloud Tenant ID"
    )
    chroma_database: str = Field(
        default="gfg_rag",
        description="Target database housing the indexed chunks"
    )
    chroma_collection: str = Field(
        default="articles",
        description="Collection name storing 3.4 Lakh GFG articles"
    )

    # Primary LLM Provider Choice
    primary_provider: Literal["groq", "google"] = Field(
        default="groq",
        description="Primary LLM provider to attempt first (groq or google)"
    )

    # Google Gemini Configuration
    gemini_api_key: str = Field(
        default="",
        description="Google AI Studio API Key for Gemini"
    )
    primary_llm_model: str = Field(
        default="gemini-3.8-flash",
        description="Primary fast & lean generation model"
    )

    # Fallback LLM: Groq (Bypasses 429/503 spikes)
    groq_api_key: str = Field(
        default="",
        description="Groq API key for LPU failover"
    )
    fallback_llm_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Fallback high-speed LPU model"
    )

    # Retrieval Hyperparameters
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of relevant GFG chunks to retrieve per query"
    )

    # Server Configuration (Render Web Service)
    host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    port: int = Field(default=8000, description="FastAPI bind port")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of application settings.
    
    Using lru_cache prevents re-reading disk and re-parsing .env
    on every incoming API request.
    """
    return Settings()
