"""RAG Triad Automated Evaluation Engine.

Evaluates the production RAG pipeline against the industry standard RAG Triad:
  1. Context Relevance: Does the retrieved GFG context contain the needed technical facts?
  2. Groundedness (Faithfulness): Is the answer 100% supported by the context without hallucination?
  3. Answer Relevance: Does the response directly address the user's technical inquiry?

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --samples 3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Windows UTF-8 Terminal Fix
sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.config import get_settings
from src.services.generator import get_generator_service
from src.services.retriever import get_retriever_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Standard Computer Science & DSA Golden Evaluation Benchmark
GOLDEN_DATASET = [
    {
        "query": "What is binary search and what is its time complexity?",
        "expected_keywords": ["divide", "sorted", "O(log n)", "middle", "half"],
        "category": "Searching Algorithms",
    },
    {
        "query": "Explain Dynamic Programming and its two key properties.",
        "expected_keywords": ["optimal substructure", "overlapping subproblems", "memoization", "tabulation"],
        "category": "Algorithm Design",
    },
    {
        "query": "What is the difference between a process and a thread in OS?",
        "expected_keywords": ["memory", "address space", "lightweight", "context switch", "shared"],
        "category": "Operating Systems",
    },
    {
        "query": "How does Dijkstra's algorithm find the shortest path in a graph?",
        "expected_keywords": ["greedy", "priority queue", "min-heap", "non-negative", "relaxation"],
        "category": "Graph Theory",
    },
]


def evaluate_context_relevance(context: str, expected_keywords: list[str]) -> float:
    """Measures what fraction of essential topic keywords are present in retrieved context."""
    if not context or "No relevant context found" in context:
        return 0.0
    text_lower = context.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return round(matches / len(expected_keywords), 2)


def evaluate_faithfulness(answer: str, context: str) -> float:
    """Heuristic check: Verifies key factual assertions in answer are grounded in context."""
    if not answer or "cannot find enough details" in answer:
        return 1.0  # Safe rejection is 100% faithful to context limits
    
    # Check overlap of substantial content words
    answer_words = set(w.lower() for w in answer.split() if len(w) > 4)
    context_words = set(w.lower() for w in context.split() if len(w) > 4)
    
    if not answer_words:
        return 1.0
    
    grounded_words = answer_words.intersection(context_words)
    score = min(1.0, (len(grounded_words) / len(answer_words)) * 1.3)
    return round(score, 2)


def evaluate_answer_relevance(query: str, answer: str, expected_keywords: list[str]) -> float:
    """Measures how well the answer addresses the specific technical query."""
    if not answer:
        return 0.0
    
    ans_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    return round(matches / len(expected_keywords), 2)


def run_rag_triad_evaluation(samples: int | None = None) -> None:
    """Runs end-to-end evaluation across the golden dataset and prints a metric scorecard."""
    test_cases = GOLDEN_DATASET[:samples] if samples else GOLDEN_DATASET
    total_cases = len(test_cases)

    logger.info("=" * 70)
    logger.info("  🚀 Starting GFG RAG Triad Evaluation Suite")
    logger.info(f"  Total Benchmark Test Cases: {total_cases}")
    logger.info("=" * 70)

    retriever = get_retriever_service()
    generator = get_generator_service()

    results = []
    total_start = time.perf_counter()

    for idx, tc in enumerate(test_cases, 1):
        query = tc["query"]
        category = tc["category"]
        keywords = tc["expected_keywords"]

        logger.info(f"\n[{idx}/{total_cases}] Evaluating: '{query}' ({category})")
        t0 = time.perf_counter()

        # 1. Retrieve
        chunks = retriever.retrieve(query=query, top_k=4)
        context = retriever.format_context(chunks)

        # 2. Generate
        gen_result = generator.generate(query=query, context=context)
        latency = time.perf_counter() - t0

        # 3. Calculate RAG Triad Metrics
        c_rel = evaluate_context_relevance(context, keywords)
        faith = evaluate_faithfulness(gen_result.answer, context)
        a_rel = evaluate_answer_relevance(query, gen_result.answer, keywords)
        composite = round((c_rel + faith + a_rel) / 3, 2)

        results.append({
            "query": query,
            "category": category,
            "provider": gen_result.provider.upper(),
            "context_relevance": c_rel,
            "faithfulness": faith,
            "answer_relevance": a_rel,
            "composite_score": composite,
            "latency_s": round(latency, 2),
        })

        logger.info(
            f"    -> Context Rel: {c_rel:.2f} | Faithfulness: {faith:.2f} | "
            f"Answer Rel: {a_rel:.2f} | Composite: {composite:.2f} | Latency: {latency:.2f}s"
        )

    total_time = time.perf_counter() - total_start

    # Summary Scorecard
    avg_c_rel = round(sum(r["context_relevance"] for r in results) / total_cases, 2)
    avg_faith = round(sum(r["faithfulness"] for r in results) / total_cases, 2)
    avg_a_rel = round(sum(r["answer_relevance"] for r in results) / total_cases, 2)
    avg_composite = round(sum(r["composite_score"] for r in results) / total_cases, 2)
    avg_latency = round(sum(r["latency_s"] for r in results) / total_cases, 2)

    print("\n" + "=" * 70)
    print("🏆 GFG RAG TRIAD EVALUATION SCORECARD")
    print("=" * 70)
    print(f"{'Category':<22} | {'C-Rel':<6} | {'Faith':<6} | {'A-Rel':<6} | {'Score':<6} | {'Latency'}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['category']:<22} | {r['context_relevance']:<6.2f} | {r['faithfulness']:<6.2f} | "
            f"{r['answer_relevance']:<6.2f} | {r['composite_score']:<6.2f} | {r['latency_s']:.2f}s"
        )
    print("-" * 70)
    print(
        f"{'AVERAGE / OVERALL':<22} | {avg_c_rel:<6.2f} | {avg_faith:<6.2f} | "
        f"{avg_a_rel:<6.2f} | {avg_composite:<6.2f} | {avg_latency:.2f}s"
    )
    print("=" * 70)
    print(f"Total Evaluation Time: {total_time:.2f}s across {total_cases} queries.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GFG RAG Triad Evaluation Runner")
    parser.add_argument("--samples", type=int, default=None, help="Number of benchmark test cases to run")
    args = parser.parse_args()

    run_rag_triad_evaluation(samples=args.samples)


if __name__ == "__main__":
    main()

