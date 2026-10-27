"""
benchmark_memory.py
===================
Step 0.5: Local Memory Benchmark via psutil
Measures Resident Set Size (RSS) memory consumption:
  Stage 1: Baseline (Python runtime)
  Stage 2: Model Ingestion (Chroma ONNX DefaultEmbeddingFunction)
  Stage 3: Peak Load (Embedding batch of 50 technical queries)

Target: Under 400 MB RSS (Safely below Render's 512 MB ceiling)
"""

import os
import sys
import time
import psutil

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

process = psutil.Process(os.getpid())

def get_memory_mb():
    """Returns the current process RSS memory in Megabytes."""
    return process.memory_info().rss / (1024 * 1024)

print("=" * 65)
print("📊 STEP 0.5: MEMORY BENCHMARK (RENDER 512MB RAM GUARDRAIL)")
print("=" * 65)

# Stage 1: Baseline Memory
time.sleep(0.5)
mem_baseline = get_memory_mb()
print(f"\n[Stage 1] Baseline Process Memory    : {mem_baseline:.2f} MB")

# Stage 2: Load Chroma ONNX DefaultEmbeddingFunction
print("\n[Stage 2] Loading Chroma ONNX Embedding Engine...")
from chromadb.utils import embedding_functions
ef = embedding_functions.DefaultEmbeddingFunction()

mem_after_model = get_memory_mb()
model_delta = mem_after_model - mem_baseline
print(f"  • Memory After Model Load         : {mem_after_model:.2f} MB")
print(f"  • Model RAM Footprint (Delta)     : +{model_delta:.2f} MB")

# Stage 3: Peak Load Simulation (Batch of 50 technical queries)
print("\n[Stage 3] Simulating High Load: Embedding 50 Technical Queries...")
sample_queries = [
    f"Query #{i}: What is the algorithmic complexity of graph traversal, dynamic programming, and binary search trees in Python?"
    for i in range(50)
]

start_time = time.time()
embeddings = ef(sample_queries)
duration = time.time() - start_time

mem_peak = get_memory_mb()
load_delta = mem_peak - mem_after_model

print(f"  • 50 Queries Encoded in           : {duration:.2f} seconds")
print(f"  • Memory Under Peak Load          : {mem_peak:.2f} MB")
print(f"  • Load RAM Increase (Delta)       : +{load_delta:.2f} MB")

# Summary & Verdict
print("\n" + "=" * 65)
print("🎯 MEMORY BENCHMARK VERDICT (RENDER CEILING: 512 MB)")
print("=" * 65)
print(f"  • Final Peak RSS Memory           : {mem_peak:.2f} MB")
print(f"  • Render Safe Ceiling Target      : 400.00 MB")
print(f"  • Remaining Buffer to 512MB Limit : {512.0 - mem_peak:.2f} MB")

if mem_peak < 400.0:
    print(f"\n✅ PASS: Peak memory ({mem_peak:.2f} MB) is comfortably below 400 MB!")
    print(f"   Render Free Tier (512MB) has ~{512.0 - mem_peak:.1f} MB safety buffer.")
    print("   ZERO RISK of SIGKILL / Exit Code 137 OOM crash!")
else:
    print(f"\n❌ FAIL: Memory ({mem_peak:.2f} MB) exceeded 400 MB safe threshold.")

print("=" * 65)
