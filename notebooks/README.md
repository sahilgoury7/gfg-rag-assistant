# 📓 GeeksforGeeks Advanced RAG Pipeline — Research & Prototyping Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahilgoury7/gfg-rag-assistant/blob/main/notebooks/GeeksforGeeks_Advanced_RAG_Pipeline.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ChromaDB Cloud](https://img.shields.io/badge/ChromaDB-3.4L%20Chunks-orange.svg)](https://trychroma.com)

This directory contains the original, end-to-end interactive Jupyter notebook used to design, benchmark, and evaluate the **GeeksforGeeks RAG pipeline** before it was transformed into our production FastAPI microservice and Streamlit Cloud web application.

---

## 🎯 Notebook Overview

* **File:** [`GeeksforGeeks_Advanced_RAG_Pipeline.ipynb`](GeeksforGeeks_Advanced_RAG_Pipeline.ipynb)
* **Knowledge Base Corpus:** **49,298 verified GeeksforGeeks articles**
* **Indexed Vectors:** **340,153 semantic chunks** hosted in ChromaDB Cloud
* **Core Technologies:** LangChain Text Splitters, ChromaDB Cloud, SBERT / FastEmbed (384-d), BM25 Lexical Search, Reciprocal Rank Fusion (RRF), Cross-Encoder Re-Ranking, and Groq/Gemini generation.

---

## 🔬 Pipeline Architecture in the Notebook

The notebook walks through the complete 7-stage RAG lifecycle:

```
[49,298 GFG Articles] 
        │
        ▼ (RecursiveCharacterTextSplitter: 1200 chars, 200 overlap)
[340,153 Semantic Chunks]
        │
        ├──> [ChromaDB Cloud Vector Collection] (Batch Size = 250)
        │
        ▼
[User Technical Query]
        │
        ├──> Dense Vector Search (ChromaDB HNSW Cosine Similarity)
        ├──> Lexical Keyword Search (BM25Okapi)
        │
        ▼
[Reciprocal Rank Fusion (RRF)] ➔ Merges Dense + Sparse Ranks
        │
        ▼
[Cross-Encoder Re-Ranking] ➔ Joint Cross-Attention Scoring (`ms-marco-MiniLM-L-6-v2`)
        │
        ▼
[Top-3 High-Precision Context Chunks]
        │
        ▼
[Dual-LLM Generation Engine] ➔ Groq LPU (Primary) + Google Gemini (Fallback)
        │
        ▼
[Grounded Answer with Source Links & Zero Hallucination]
```

---

## 🚀 How to Run in Google Colab

### 1. Launch with One Click
Click the badge at the top or open:
👉 **[Open in Google Colab](https://colab.research.google.com/github/sahilgoury7/gfg-rag-assistant/blob/main/notebooks/GeeksforGeeks_Advanced_RAG_Pipeline.ipynb)**

### 2. Runtime Selection
* **Hardware Accelerator:** Standard CPU or T4 GPU (GPU is recommended for faster initial local embeddings if re-indexing).

### 3. Required Secrets (Colab Keys 🔑)
Add the following keys in Colab's left sidebar under **Secrets** (`🔑` icon):
| Secret Name | Description | Where to Get |
| :--- | :--- | :--- |
| `CHROMA_API_KEY` | Chroma Cloud authentication token | [trychroma.com](https://trychroma.com) |
| `GROQ_API_KEY` | Ultra-fast Groq LPU inference key | [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | Google AI Studio key for fallback generation | [aistudio.google.com](https://aistudio.google.com) |

---

## 🏭 From Colab Prototype to Production Microservice

While this notebook proved the viability of hybrid search and re-ranking, running interactive notebooks is not viable for production. Here is how this notebook was translated into the repository's production code:

| Notebook Prototype (`notebooks/`) | Production Microservice (`src/` & Cloud) |
| :--- | :--- |
| Global variables in memory (`chunked_data`, `df_dict`) | Stateless REST API with Pydantic v2 validation (`src/api/schemas.py`) |
| Heavy PyTorch embeddings (~1.2 GB RAM bloat) | Lightweight C++ ONNX runtime strictly holding **105 MB RAM** (`src/services/embedding.py`) |
| Interactive shift-enter execution | Docker container deployed on Render Cloud (`Dockerfile`) |
| Local display output in notebook cells | Streamlit Cloud live chat interface with typewriter streaming (`src/frontend/app.py`) |
| Single-point-of-failure API calls | Dual-LLM Circuit Breaker failover engine (`src/services/generator.py`) |
| Unprotected endpoints | IP-based rate limiting via Slowapi at 15 req/min (`src/core/limiter.py`) |
| Manual cell checking | Automated 17-test Pytest suite & GitHub Actions CI (`tests/` & `.github/`) |

---

## 📊 Summary Dataset Statistics
* **Raw Articles Count:** `49,298`
* **Text Splitter Settings:** `chunk_size = 1200`, `chunk_overlap = 200`
* **Total Chunks Produced:** `340,153`
* **Batch Ingestion Size:** `250 chunks per upsert`
* **Chroma Cloud Collection Name:** `articles`
