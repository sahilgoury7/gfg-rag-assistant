# 🧠 GFG RAG Assistant (Production RAG System)

A production-ready Retrieval-Augmented Generation (RAG) assistant indexing **3.4 Lakh GeeksforGeeks articles** using Chroma Cloud, FastAPI, and Streamlit.

---

## 🏗️ Architecture Overview

- **Vector Database**: Chroma Cloud (3.4 Lakh Chunks, HNSW Index)
- **Embeddings**: Chroma built-in `DefaultEmbeddingFunction` (ONNX Runtime, `all-MiniLM-L6-v2`, 384-dim, < 120MB RAM)
- **Backend API**: FastAPI (REST + SSE Streaming, Pydantic v2 schemas)
- **Frontend UI**: Streamlit (Chat UI with clickable GFG source citations & cold-start health monitoring)
- **LLM Engine**: Primary Gemini 2.5 Flash + Failover Groq LLaMA 3.3 70B
- **Deployment**: Render Native Web Service (Backend, 512MB RAM cap) + Streamlit Community Cloud (Frontend)

---

## 📁 Repository Structure

```text
gfg-rag-assistant/
│
├── src/                         # Backend (FastAPI)
│   ├── api/                     # API routes & schemas
│   ├── core/                    # Config & environment settings
│   ├── services/                # Embedding, Retriever & Generator services
│   └── main.py                  # FastAPI app entrypoint
│
├── ui/                          # Frontend (Streamlit)
│   ├── app.py                   # Main chat interface
│   └── components/              # Cold-start handler & citation widgets
│
├── scripts/                     # Ingestion, benchmarks & verification
│   ├── benchmark_memory.py      # Real RAM profiler (verified at ~105MB peak)
│   └── verify_embeddings.py     # ONNX embedding 384-dim & cosine check
│
├── .env.example                 # Environment variables template
├── .gitignore                   # Git exclusion rules
├── requirements.txt             # Lean production dependencies (<512MB RAM)
└── README.md                    # Project documentation
```

---

## 🚦 Milestone Progress

- [x] **Phase 0: Security & Memory Guardrails**
  - Scrubbed raw credentials & committed strict `.gitignore`
  - ONNX embedding engine verified without heavy PyTorch
  - Real RAM benchmark verified at **105.55 MB** via `psutil`
- [x] **Phase 1: Modular Core Architecture**
  - Fail-fast configuration with Pydantic v2 `BaseSettings`
  - Dedicated memory-safe `EmbeddingService` (384-dim)
  - Chroma Cloud `RetrieverService` querying 3.4 Lakh GFG articles with live canonical URL citations
  - Dual-LLM `GeneratorService` (Gemini 3.8 Flash primary + Groq LPU failover)
  - Terminal CLI smoke test verified end-to-end (`python scripts/test_cli.py`)
- [ ] **Phase 2: Lean FastAPI REST Backend** (Up Next)

---

## 🛠️ Quickstart (Local Development)

```bash
# 1. Clone repository
git clone https://github.com/sahilgoury7/gfg-rag-assistant.git
cd gfg-rag-assistant

# 2. Set up virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify embedding engine & memory benchmark
python scripts/verify_embeddings.py
python scripts/benchmark_memory.py
```
