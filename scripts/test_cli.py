"""Terminal CLI Smoke Test for GFG RAG Assistant.

Executes an end-to-end question-answering workflow directly from the terminal
without requiring a FastAPI web server or Streamlit UI.
Tests: Configuration -> ONNX Embedding -> Chroma Cloud Retrieval -> LLM Generation.
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.core.config import get_settings
from src.services.generator import get_generator_service
from src.services.retriever import get_retriever_service


def run_smoke_test(query: str, top_k: int = 4) -> None:
    """Runs a complete end-to-end retrieval and generation pipeline."""
    print("=" * 70)
    print("🧠 GFG RAG ASSISTANT — CLI SMOKE TEST")
    print("=" * 70)
    print(f"❓ Query: \"{query}\"")
    print(f"⚙️  Top-K: {top_k}")
    print("-" * 70)

    # 1. Validate Settings
    t_start_total = time.perf_counter()
    try:
        settings = get_settings()
        print(f"✅ Settings Loaded: Env={settings.app_env}, Primary Provider={settings.primary_provider.upper()}")
    except Exception as err:
        print(f"❌ Configuration Error: {err}")
        print("💡 Hint: Ensure .env file exists and contains valid API keys.")
        sys.exit(1)

    # 2. Retrieve Relevant Chunks from Chroma Cloud
    print("\n🔍 Step 1: Querying Chroma Cloud (3.4 Lakh Chunks)...")
    t0_retrieval = time.perf_counter()
    try:
        retriever = get_retriever_service()
        chunks = retriever.retrieve(query=query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0_retrieval) * 1000
        print(f"✅ Successfully retrieved {len(chunks)} chunks in {retrieval_ms:.1f} ms.")
    except Exception as err:
        print(f"❌ Retrieval Failed: {err}")
        sys.exit(1)

    # Display Retrieved Chunks
    print("\n📚 Retrieved Citations:")
    for idx, chunk in enumerate(chunks, 1):
        print(f"  [{idx}] {chunk.title}")
        print(f"      Relevance: {chunk.relevance_score * 100:.1f}% | Distance: {chunk.distance:.4f}")
        print(f"      URL: {chunk.source_url or 'N/A'}")
        preview = chunk.text.replace("\n", " ")[:120]
        print(f"      Snippet: {preview}...\n")

    # 3. Format Context & Generate Answer
    context_text = retriever.format_context(chunks)
    print("🤖 Step 2: Generating Answer via LLM Engine...")
    t0_gen = time.perf_counter()

    try:
        generator = get_generator_service()
        result = generator.generate(query=query, context=context_text)
        generation_ms = (time.perf_counter() - t0_gen) * 1000
    except Exception as err:
        print(f"❌ Generation Failed: {err}")
        sys.exit(1)

    total_ms = (time.perf_counter() - t_start_total) * 1000

    print("\n" + "=" * 70)
    print(f"🎉 FINAL ANSWER (Model: {result.model_used} | Provider: {result.provider.upper()})")
    if result.fallback_triggered:
        print("⚠️  [NOTICE: Fallback was triggered due to primary failure]")
    print("=" * 70)
    print(result.answer)
    print("=" * 70)

    # Latency Telemetry Dashboard
    print("\n⏱️  LATENCY TELEMETRY BENCHMARK:")
    print(f"  • Chroma Cloud Retrieval (3.4L Chunks): {retrieval_ms:.1f} ms")
    print(f"  • LLM Generation ({result.provider.upper()}):          {generation_ms:.1f} ms")
    print(f"  • Total End-to-End Turnaround:          {total_ms:.1f} ms ({total_ms / 1000:.2f} s)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="GFG RAG CLI Smoke Test")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the time complexity of Binary Search?",
        help="Question to ask the RAG assistant"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)"
    )
    args = parser.parse_args()
    run_smoke_test(query=args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
