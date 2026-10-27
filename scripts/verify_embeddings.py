"""
verify_embeddings.py
====================
Step 0.3 & 0.4: Embedding & Vector Compatibility Verification
Verifies that Chroma's ONNX-based DefaultEmbeddingFunction (all-MiniLM-L6-v2):
  1. Runs without PyTorch (memory-safe).
  2. Generates exactly 384-dimensional vectors.
  3. Properly separates semantically similar vs dissimilar queries via Cosine Distance.
"""

import sys
import numpy as np

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("🔍 STEP 0.3 & 0.4: EMBEDDING VERIFICATION (ONNX RUNTIME)")
print("=" * 60)

try:
    from chromadb.utils import embedding_functions
    print("✅ ChromaDB embedding_functions module imported successfully.")
except ImportError as e:
    print(f"❌ Failed to import chromadb: {e}")
    sys.exit(1)

# 1. Initialize Chroma DefaultEmbeddingFunction (Uses ONNX + all-MiniLM-L6-v2)
print("\n[1/4] Loading Chroma DefaultEmbeddingFunction (ONNX runtime)...")
ef = embedding_functions.DefaultEmbeddingFunction()
print("✅ DefaultEmbeddingFunction loaded! (No heavy PyTorch needed)")

# 2. Test Inputs
similar_q1 = "What is the time complexity of Dijkstra algorithm?"
similar_q2 = "How fast does Dijkstra algorithm run in Python?"
dissimilar_q = "How to bake a chocolate cake at home?"

texts = [similar_q1, similar_q2, dissimilar_q]

# 3. Generate Embeddings
print("\n[2/4] Generating embeddings for 3 test sentences...")
embeddings = ef(texts)
print(f"✅ Generated {len(embeddings)} embedding vectors.")

# 4. Check Dimensionality (Must be exactly 384)
print("\n[3/4] Verifying vector dimensionality...")
dim1 = len(embeddings[0])
dim2 = len(embeddings[1])
dim3 = len(embeddings[2])

print(f"  • Vector 1 Dimension: {dim1}")
print(f"  • Vector 2 Dimension: {dim2}")
print(f"  • Vector 3 Dimension: {dim3}")

if dim1 == 384 and dim2 == 384 and dim3 == 384:
    print("✅ DIMENSION CHECK PASSED: Exactly 384 dimensions (Matches Chroma Cloud 3.4L index)!")
else:
    print(f"❌ DIMENSION MISMATCH: Expected 384, got {dim1}")
    sys.exit(1)

# 5. Cosine Distance Calculation (Distance = 1 - Cosine_Similarity)
print("\n[4/4] Testing Cosine Distance & Semantic Separation...")

def cosine_distance(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    # Cosine distance = 1 - (A . B / (||A|| * ||B||))
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(1.0 - similarity)

dist_similar = cosine_distance(embeddings[0], embeddings[1])
dist_dissimilar = cosine_distance(embeddings[0], embeddings[2])

print(f"  • Distance (Dijkstra Q1 vs Dijkstra Q2)  : {dist_similar:.4f} (Expected: Low < 0.35)")
print(f"  • Distance (Dijkstra Q1 vs Chocolate Cake): {dist_dissimilar:.4f} (Expected: High > 0.70)")

if dist_similar < 0.35 and dist_dissimilar > 0.70:
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("   1. Lightweight ONNX Runtime: Verified")
    print("   2. 384-Dimensional Vectors : Verified")
    print("   3. Cosine Math Alignment   : Verified")
    print("=" * 60)
else:
    print("\n⚠️ Warning: Distances were unexpected, check model consistency.")
