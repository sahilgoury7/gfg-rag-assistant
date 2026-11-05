"""Unit Tests for Generator Service & Dual-LLM Circuit Breaker.

Validates:
- Primary LLM (Groq) success execution
- Automatic Failover to Gemini when Primary encounters errors (429/503/timeout)
- RuntimeError raise when both providers fail
"""

from unittest.mock import MagicMock
import pytest

from src.services.generator import GeneratorService


def test_generator_primary_success():
    """Generator should return response from primary provider when it succeeds."""
    mock_settings = MagicMock()
    mock_settings.primary_provider = "groq"
    mock_settings.fallback_llm_model = "llama-3.3-70b-versatile"
    mock_settings.primary_llm_model = "gemini-2.0-flash"

    generator = GeneratorService(settings=mock_settings)
    generator._generate_with_groq = MagicMock(return_value="Binary search is O(log n).")
    generator._generate_with_gemini = MagicMock()

    result = generator.generate(query="What is binary search?", context="Binary search divides space.")

    assert result.answer == "Binary search is O(log n)."
    assert result.provider == "groq"
    assert result.fallback_triggered is False
    generator._generate_with_gemini.assert_not_called()


def test_generator_fallback_triggered_on_primary_failure():
    """Generator should transparently failover to Gemini when Groq fails."""
    mock_settings = MagicMock()
    mock_settings.primary_provider = "groq"
    mock_settings.fallback_llm_model = "llama-3.3-70b-versatile"
    mock_settings.primary_llm_model = "gemini-2.0-flash"

    generator = GeneratorService(settings=mock_settings)
    generator._generate_with_groq = MagicMock(side_effect=Exception("HTTP 429: Rate limit exceeded"))
    generator._generate_with_gemini = MagicMock(return_value="Grounded answer from Gemini fallback.")

    result = generator.generate(query="Explain DP", context="Dynamic programming memoizes states.")

    assert result.answer == "Grounded answer from Gemini fallback."
    assert result.provider == "google"
    assert result.fallback_triggered is True
    generator._generate_with_groq.assert_called_once()
    generator._generate_with_gemini.assert_called_once()


def test_generator_both_providers_fail():
    """Generator should raise RuntimeError when both primary and secondary fail."""
    mock_settings = MagicMock()
    mock_settings.primary_provider = "groq"
    mock_settings.fallback_llm_model = "llama-3.3-70b-versatile"
    mock_settings.primary_llm_model = "gemini-2.0-flash"

    generator = GeneratorService(settings=mock_settings)
    generator._generate_with_groq = MagicMock(side_effect=Exception("Groq Down"))
    generator._generate_with_gemini = MagicMock(side_effect=Exception("Gemini Overloaded"))

    with pytest.raises(RuntimeError) as exc_info:
        generator.generate(query="What is quicksort?", context="Quicksort partitions.")

    assert "All LLM providers exhausted" in str(exc_info.value)
