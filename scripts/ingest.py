"""GeeksforGeeks Knowledge Base Ingestion Pipeline.

Transforms raw GeeksforGeeks articles into semantic chunks and indexes them
into ChromaDB Cloud with metadata for verifiable source citations.

Pipeline Steps:
  1. Load raw GeeksforGeeks articles (49,298 total articles).
  2. Split articles using Recursive Character Chunking (chunk_size=1200, overlap=200).
  3. Extract canonical metadata (title, article_id, tags, canonical URL).
  4. Batch-embed and upload to ChromaDB Cloud in chunks of 250 with error recovery.

Usage:
  python scripts/ingest.py --dry-run
  python scripts/ingest.py --limit 100 --batch-size 250
  python scripts/ingest.py --source-path ./data/df_dict.json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def recursive_split_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[str]:
    """Lightweight pure-Python recursive text splitter.

    Mimics LangChain's RecursiveCharacterTextSplitter with zero external dependency.
    Splits by paragraphs (\n\n), then lines (\n), then spaces ( ), preserving semantic integrity.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", " ", ""]
    
    def _split(s: str, seps: list[str]) -> list[str]:
        if not s.strip():
            return []
        if len(s) <= chunk_size:
            return [s.strip()]

        sep = seps[0] if seps else ""
        next_seps = seps[1:] if len(seps) > 1 else [""]

        splits = s.split(sep) if sep else list(s)
        chunks: list[str] = []
        current = ""

        for part in splits:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > chunk_size and next_seps:
                    chunks.extend(_split(part, next_seps))
                    current = ""
                else:
                    current = part

        if current and current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = _split(text, separators)
    
    # Apply sliding window overlap if requested
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and len(raw_chunks[i - 1]) >= chunk_overlap:
            overlap_prefix = raw_chunks[i - 1][-chunk_overlap:]
            chunk = f"{overlap_prefix} {chunk}"
        overlapped.append(chunk)

    return overlapped


def chunk_articles(
    articles_data: dict[str, Any] | list[dict[str, Any]],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Iterates through articles and segments them into semantic chunks with metadata."""
    logger.info("Starting article chunking process...")
    start_time = time.perf_counter()

    items = articles_data.items() if isinstance(articles_data, dict) else enumerate(articles_data)
    total_processed = 0
    all_chunks: list[dict[str, Any]] = []

    for key, article in items:
        if limit and total_processed >= limit:
            break

        article_id = str(article.get("article_id") or key)
        title = article.get("title") or "GeeksforGeeks Article"
        link = article.get("link") or article.get("url") or ""
        tags = article.get("tags") or ""
        article_text = str(article.get("article") or article.get("text") or "")

        if not article_text or article_text.lower() == "nan":
            continue

        text_chunks = recursive_split_text(
            article_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for idx, chunk_text in enumerate(text_chunks, 1):
            all_chunks.append({
                "chunk_id": f"{article_id}_chunk_{idx}",
                "article_id": article_id,
                "title": title,
                "link": link,
                "tags": str(tags),
                "text": chunk_text,
            })

        total_processed += 1

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Chunking complete: {total_processed} articles -> {len(all_chunks)} chunks in {elapsed:.2f}s"
    )
    return all_chunks


def upload_batches_to_chroma(
    chunks: list[dict[str, Any]],
    batch_size: int = 250,
    dry_run: bool = False,
) -> None:
    """Pushes chunks into ChromaDB Cloud in batches of 250 with automatic retry."""
    if dry_run:
        logger.info(f"[DRY-RUN] Verified {len(chunks)} chunks. Skipping remote Chroma Cloud upload.")
        return

    settings = get_settings()
    import chromadb

    logger.info(f"Connecting to ChromaDB Cloud (Database: {settings.chroma_database})...")
    client = chromadb.CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )
    collection = client.get_or_create_collection(name=settings.chroma_collection)
    initial_count = collection.count()
    logger.info(f"Target collection '{settings.chroma_collection}' current count: {initial_count}")

    total_chunks = len(chunks)
    logger.info(f"Beginning upload of {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "article_id": c["article_id"],
                "title": c["title"],
                "link": c["link"],
                "tags": c["tags"],
            }
            for c in batch
        ]

        # Automatic retry logic for resilient cloud batching
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(f"Uploaded batch [{i + len(batch)}/{total_chunks}] successfully.")
                break
            except Exception as exc:
                if attempt == retries:
                    logger.error(f"Failed to upload batch {i}-{i+len(batch)} after {retries} attempts: {exc}")
                    raise
                logger.warning(f"Batch {i} failed attempt {attempt}/{retries}: {exc}. Retrying in 2s...")
                time.sleep(2)

    final_count = collection.count()
    logger.info(f"✅ Ingestion finished! Chroma Cloud now contains {final_count} documents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GFG RAG Knowledge Base Ingestion Pipeline")
    parser.add_argument("--source-path", type=str, default="", help="Path to raw articles JSON file")
    parser.add_argument("--batch-size", type=int, default=250, help="Batch size for Chroma upload (default: 250)")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Character size per chunk (default: 1200)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap characters between chunks (default: 200)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles to process (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Process and validate chunks without uploading to Chroma")

    args = parser.parse_args()

    # If no source file provided, demonstrate with mock sample to verify pipeline integrity
    if not args.source_path or not Path(args.source_path).exists():
        logger.info("No external source file provided. Running demonstration with sample DSA articles.")
        sample_articles = {
            "art_001": {
                "title": "Binary Search Algorithm",
                "link": "https://www.geeksforgeeks.org/binary-search/",
                "tags": "dsa, algorithms, searching",
                "article": (
                    "Binary Search is a searching algorithm used in a sorted array by repeatedly dividing the search "
                    "interval in half. The idea of binary search is to use the information that the array is sorted "
                    "and reduce the time complexity to O(log N). At each step, compare the target with the middle element. "
                    "If the middle element matches, return the index. If smaller, search the left half. If larger, search the right."
                ),
            },
            "art_002": {
                "title": "Dynamic Programming Overview",
                "link": "https://www.geeksforgeeks.org/dynamic-programming/",
                "tags": "dsa, algorithms, dp",
                "article": (
                    "Dynamic Programming (DP) is an algorithmic technique for solving optimization problems by breaking "
                    "them down into simpler subproblems and utilizing the fact that the optimal solution to the overall "
                    "problem depends upon the optimal solution to its subproblems. Two essential properties are Overlapping "
                    "Subproblems and Optimal Substructure. DP avoids redundant calculations using Memoization (Top-Down) "
                    "or Tabulation (Bottom-Up)."
                ),
            },
        }
        raw_data = sample_articles
    else:
        logger.info(f"Loading raw articles from {args.source_path}...")
        with open(args.source_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    chunks = chunk_articles(
        raw_data,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        limit=args.limit,
    )

    upload_batches_to_chroma(
        chunks,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
