"""Phase 2 Smoke Test — FastAPI Server + Endpoints.

Tests:
  1. GET  /health        → Liveness probe
  2. POST /api/v1/query  → Full RAG pipeline via HTTP
  3. POST /api/v1/query  → Validation rejection (empty query)
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Windows UTF-8 fix
sys.stdout.reconfigure(encoding="utf-8")

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_URL = "http://127.0.0.1:8001"


def start_server() -> subprocess.Popen:
    """Starts the FastAPI server as a background subprocess."""
    print("🚀 Starting FastAPI server on port 8001...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app",
         "--host", "127.0.0.1", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    print("⏳ Waiting 10s for server boot + service warm-up...")
    time.sleep(10)
    return proc


def test_health() -> None:
    """Test 1: GET /health"""
    print("\n" + "=" * 55)
    print("TEST 1 — GET /health")
    print("=" * 55)
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as r:
        data = json.loads(r.read())
    print(f"  Status      : {data['status']}")
    print(f"  Version     : {data['version']}")
    print(f"  Environment : {data['environment']}")
    assert data["status"] == "ok", "Health check failed!"
    print("  ✅ PASSED")


def test_query() -> None:
    """Test 2: POST /api/v1/query — valid request"""
    print("\n" + "=" * 55)
    print("TEST 2 — POST /api/v1/query (valid)")
    print("=" * 55)
    payload = json.dumps(
        {"query": "What is dynamic programming?", "top_k": 3}
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())

    print(f"  Provider      : {data['provider_used']}")
    print(f"  Fallback Used : {data['fallback_triggered']}")
    print(f"  Latency       : {data['latency_ms']} ms")
    print(f"  Citations     : {len(data['citations'])} sources")
    for c in data["citations"]:
        print(f"    - [{c['relevance_score']:.3f}] {c['title']}")
    print(f"  Answer Preview: {data['answer'][:150]}...")
    assert data["answer"], "Answer must not be empty!"
    print("  ✅ PASSED")


def test_validation_rejection() -> None:
    """Test 3: POST /api/v1/query — should reject empty query with 422"""
    print("\n" + "=" * 55)
    print("TEST 3 — POST /api/v1/query (invalid: empty query)")
    print("=" * 55)
    payload = json.dumps({"query": "  ", "top_k": 3}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print("  ❌ FAILED — Server should have rejected empty query!")
    except urllib.error.HTTPError as e:
        if e.code == 422:
            print(f"  HTTP 422 Unprocessable Entity — Pydantic rejected the request ✅")
            print("  ✅ PASSED")
        else:
            print(f"  ❌ Unexpected HTTP {e.code}")


def main() -> None:
    proc = start_server()
    try:
        test_health()
        test_query()
        test_validation_rejection()
        print("\n" + "=" * 55)
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("=" * 55)
        print(f"\n📖 Swagger UI: {BASE_URL}/docs")
        print(f"📄 ReDoc UI  : {BASE_URL}/redoc")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    finally:
        proc.terminate()
        print("\n🛑 Server stopped.")


if __name__ == "__main__":
    main()
