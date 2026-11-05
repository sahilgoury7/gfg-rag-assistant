"""Unit Tests for Application Configuration (config.py).

Validates:
- Settings parsing from environment or explicit values
- top_k bounds enforcement (ge=1, le=10)
- Provider choices validation
"""

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_settings_valid_creation():
    """Settings should correctly parse provided environment parameters."""
    s = Settings(
        chroma_api_key="test_chroma_key",
        groq_api_key="test_groq_key",
        primary_provider="groq",
        top_k=5,
    )
    assert s.chroma_api_key == "test_chroma_key"
    assert s.groq_api_key == "test_groq_key"
    assert s.primary_provider == "groq"
    assert s.top_k == 5


def test_settings_top_k_bounds():
    """top_k must enforce 1 <= top_k <= 10."""
    with pytest.raises(ValidationError):
        Settings(chroma_api_key="k", top_k=0)

    with pytest.raises(ValidationError):
        Settings(chroma_api_key="k", top_k=15)


def test_settings_invalid_provider():
    """primary_provider must only accept 'groq' or 'google'."""
    with pytest.raises(ValidationError):
        Settings(chroma_api_key="k", primary_provider="openai")

