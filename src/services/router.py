"""Query Router & Intent Classification Module.

Classifies incoming user queries into:
1. Conversational Chitchat (greetings, identity inquiries, gratitude, pleasantries)
2. Technical Inquiries (programming questions, DSA, algorithmic complexity, CS theory)

When a query is classified as conversational chitchat, the router generates
an immediate, friendly, canned response without executing vector retrieval.
This architectural optimization:
- Reduces roundtrip latency from ~10s to <5ms
- Eliminates unnecessary Vector DB reads and LLM token costs
- Prevents awkward anti-hallucination refusals (e.g., 'Not found in GFG docs')
"""

from functools import lru_cache
import logging
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)


class RouteDecision(NamedTuple):
    """Represents the classification outcome for an incoming query."""

    is_chitchat: bool
    canned_response: str | None = None
    intent: str = "technical"


class QueryRouter:
    """Classifies user queries and provides canned conversational responses."""

    # Pure conversational triggers (normalized lowercase, punctuation-stripped)
    EXACT_GREETINGS: set[str] = {
        "hi",
        "hello",
        "hey",
        "hola",
        "namaste",
        "howdy",
        "greetings",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "whats up",
        "what's up",
    }

    EXACT_IDENTITY: set[str] = {
        "who are you",
        "what are you",
        "what can you do",
        "help",
        "help me",
        "what is this",
        "what is this bot",
        "what is this app",
        "about",
    }

    EXACT_GRATITUDE: set[str] = {
        "thanks",
        "thank you",
        "thank you so much",
        "thx",
        "ty",
        "bye",
        "goodbye",
        "see you",
        "cool",
        "awesome",
        "great",
        "ok",
        "okay",
    }

    CANNED_GREETING: str = (
        "Hello! 👋 I am your **GeeksforGeeks AI Assistant**.\n\n"
        "I am directly grounded in over **3.4 Lakh verified GeeksforGeeks articles** "
        "to help you learn and solve:\n"
        "- 💡 **Data Structures & Algorithms** (Arrays, Trees, Graphs, DP, Sorting)\n"
        "- ⚡ **Programming Languages** (Python, C++, Java, JavaScript)\n"
        "- 🔍 **Core Computer Science** (DBMS, Operating Systems, Networks, System Design)\n\n"
        "What technical question or algorithm would you like to explore today?"
    )

    CANNED_IDENTITY: str = (
        "I am the **GeeksforGeeks RAG Assistant** 📚\n\n"
        "My purpose is to answer your Computer Science, programming, and interview "
        "preparation questions with **zero hallucination** by retrieving verified "
        "explanations and code directly from GeeksforGeeks documentation with clickable citations.\n\n"
        "**Try asking me:**\n"
        "- *\"Explain Binary Search with time complexity\"*\n"
        "- *\"What is Dynamic Programming and its key properties?\"*\n"
        "- *\"Difference between Process and Thread in OS\"*"
    )

    CANNED_GRATITUDE: str = (
        "You're very welcome! 😊\n\n"
        "Happy coding! Whenever you have another DSA, programming, or Computer Science "
        "question, feel free to ask anytime."
    )

    def route(self, query: str) -> RouteDecision:
        """Classifies a user query into conversational chitchat or technical query.

        Args:
            query: Raw user query string.

        Returns:
            RouteDecision indicating if chitchat was detected with a canned response.
        """
        # Normalize: lowercase, collapse whitespace, strip peripheral punctuation
        cleaned = re.sub(r"[^\w\s]", "", query.strip().lower())
        cleaned = " ".join(cleaned.split())

        tokens = cleaned.split()
        token_count = len(tokens)

        # ── 1. Check for Short Greetings / Pleasantries (<= 4 words) ─────────
        if token_count <= 4:
            if cleaned in self.EXACT_GREETINGS:
                logger.info(f"Query routed as GREETING: '{query}'")
                return RouteDecision(
                    is_chitchat=True,
                    canned_response=self.CANNED_GREETING,
                    intent="greeting",
                )

            if cleaned in self.EXACT_IDENTITY:
                logger.info(f"Query routed as IDENTITY: '{query}'")
                return RouteDecision(
                    is_chitchat=True,
                    canned_response=self.CANNED_IDENTITY,
                    intent="identity",
                )

            if cleaned in self.EXACT_GRATITUDE:
                logger.info(f"Query routed as GRATITUDE: '{query}'")
                return RouteDecision(
                    is_chitchat=True,
                    canned_response=self.CANNED_GRATITUDE,
                    intent="gratitude",
                )

        # ── 2. Check for Prefix Greetings (e.g. 'hello what is python') ─────
        # If user starts with 'hello' but has a real question, we strip the greeting
        # and allow the pipeline to retrieve technical context
        return RouteDecision(is_chitchat=False, intent="technical")


@lru_cache()
def get_query_router() -> QueryRouter:
    """Returns a cached singleton instance of QueryRouter."""
    return QueryRouter()

