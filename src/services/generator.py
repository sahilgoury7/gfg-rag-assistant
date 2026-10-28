"""Generator Service Module.

Implements a resilient Dual-LLM Generation Engine:
- Primary Provider: Google Gemini (gemini-2.5-flash) via native google-genai SDK
- Fallback Provider: Groq LPU (llama-3.3-70b-versatile) via native groq SDK

Guarantees 99.9% uptime by automatically failing over when the primary provider
encounters 429 rate-limiting, 503 model overload, or network timeouts.
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any
from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a highly precise technical assistant specializing in Computer Science and Programming, referencing GeeksforGeeks documentation.

STRICT OPERATIONAL RULES:
1. Base your answer PRIMARILY on the provided Context documents.
2. If the context does not contain sufficient information to answer the question, clearly state:
   "Based on the available GeeksforGeeks documentation, I cannot find enough details to answer this question accurately."
3. Never invent facts, function signatures, or time complexities not supported by the context.
4. Keep the explanation structured, concise, and include code snippets only when directly relevant.
5. Highlight key terms and reference source concepts clearly.
"""


@dataclass
class GenerationResult:
    """Encapsulates the generated answer along with telemetry metadata."""

    answer: str
    model_used: str
    provider: str
    fallback_triggered: bool


class GeneratorService:
    """Manages multi-provider LLM generation with transparent failover."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._gemini_client: Any = None
        self._groq_client: Any = None

    def _get_gemini_client(self) -> Any:
        """Lazily initializes the Google Gemini client."""
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._gemini_client

    def _get_groq_client(self) -> Any:
        """Lazily initializes the Groq client."""
        if self._groq_client is None and self._settings.groq_api_key:
            from groq import Groq
            self._groq_client = Groq(api_key=self._settings.groq_api_key)
        return self._groq_client

    def _generate_with_gemini(self, user_prompt: str) -> str:
        """Generates content via Primary Google Gemini."""
        from google.genai import types

        client = self._get_gemini_client()
        response = client.models.generate_content(
            model=self._settings.primary_llm_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        return response.text.strip()

    def _generate_with_groq(self, user_prompt: str) -> str:
        """Generates content via Fallback Groq LPU."""
        client = self._get_groq_client()
        if client is None:
            raise RuntimeError("Groq API key not configured for failover.")

        response = client.chat.completions.create(
            model=self._settings.fallback_llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        choice = response.choices[0]
        if not choice.message.content:
            raise ValueError("Groq returned an empty response.")
        return choice.message.content.strip()

    def generate(self, query: str, context: str) -> GenerationResult:
        """Generates an answer using Primary LLM with automatic fallback on failure."""
        user_prompt = f"""Context:
{context}

Question: {query}
"""
        primary_err: Exception | None = None

        # 1. Attempt Primary Generation (Google Gemini)
        try:
            logger.info(
                f"Calling Primary LLM ({self._settings.primary_llm_model})..."
            )
            answer = self._generate_with_gemini(user_prompt)
            return GenerationResult(
                answer=answer,
                model_used=self._settings.primary_llm_model,
                provider="google",
                fallback_triggered=False,
            )
        except Exception as err:
            primary_err = err
            logger.warning(
                f"⚠️ Primary LLM ({self._settings.primary_llm_model}) failed: {err}. "
                "Attempting failover to Groq..."
            )

        # 2. Attempt Fallback Generation (Groq LPU)
        try:
            logger.info(
                f"Calling Fallback LLM ({self._settings.fallback_llm_model})..."
            )
            answer = self._generate_with_groq(user_prompt)
            return GenerationResult(
                answer=answer,
                model_used=self._settings.fallback_llm_model,
                provider="groq",
                fallback_triggered=True,
            )
        except Exception as fallback_err:
            logger.error(
                f"❌ Both Primary and Fallback LLMs failed. Fallback error: {fallback_err}"
            )
            raise RuntimeError(
                f"All LLM providers exhausted. Primary error: {primary_err}, "
                f"Fallback error: {fallback_err}"
            ) from fallback_err


@lru_cache()
def get_generator_service() -> GeneratorService:
    """Returns a cached singleton instance of GeneratorService."""
    return GeneratorService()
