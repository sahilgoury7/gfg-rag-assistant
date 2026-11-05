# 🧠 GFG RAG Assistant (Production-Grade RAG Engine)

[![Python Version](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB Cloud](https://img.shields.io/badge/ChromaDB-3.4L%20Chunks-orange.svg)](https://trychroma.com)
[![Live Backend](https://img.shields.io/badge/Render-Live%20API-success.svg)](https://gfg-rag-assistant.onrender.com/docs)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade, memory-efficient Retrieval-Augmented Generation (RAG) assistant indexing **3.4 Lakh GeeksforGeeks chunks** using ChromaDB Cloud, FastAPI REST backend, and Streamlit Chat UI with verifiable source citations and multi-model failover.

---

## 🌍 Live Endpoints & Interactive Demos

| Service | Environment | URL / Direct Access | Status |
| :--- | :--- | :--- | :--- |
| **Interactive API Docs** | Render Cloud | [https://gfg-rag-assistant.onrender.com/docs](https://gfg-rag-assistant.onrender.com/docs) | 🟢 **200 OK Live** |
| **API Health Probe** | Render Cloud | [https://gfg-rag-assistant.onrender.com/health](https://gfg-rag-assistant.onrender.com/health) | 🟢 **Healthy** |
| **Root Discovery** | Render Cloud | [https://gfg-rag-assistant.onrender.com/](https://gfg-rag-assistant.onrender.com/) | 🟢 **Online** |
| **Streamlit Chat UI** | Localhost / Cloud | `streamlit run src/frontend/app.py` | 🟢 **Ready** |

---

## 🏗️ Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    Streamlit Chat UI (Frontend)      │
                      │  (History, Citations, Telemetry)     │
                      └──────────────────┬───────────────────┘
                                         │ HTTP POST /api/v1/query
                                         ▼
                      ┌──────────────────────────────────────┐
                      │     FastAPI Gateway (Uvicorn)        │
                      │   - Pydantic v2 Input Validation     │
                      │   - CORS Security Middleware         │
                      │   - Lifespan Model Pre-Warming       │
                      └──────────────────┬───────────────────┘
                                         │ Threadpool Offloading
                                         ▼
                      ┌──────────────────────────────────────┐
                      │       RAG Controller Pipeline        │
                      └──────────┬────────────────┬──────────┘
                                 │                │
            1. Query Vector (384-d)               │ 3. Augmented Context
                                 ▼                ▼
     ┌─────────────────────────────┐    ┌─────────────────────────────┐
     │   EmbeddingService (ONNX)   │    │  Dual-LLM Circuit Breaker   │
     │  - all-MiniLM-L6-v2 (384-d) │    │  - Primary: Groq LPU        │
     │  - Sub-120MB RSS RAM        │    │    (openai/gpt-oss-120b)    │
     └──────────────┬──────────────┘    │  - Fallback: Google Gemini  │
                    │                   │    (gemini-3.8-flash)       │
            2. Cosine Search            └──────────────┬──────────────┘
                    ▼                                  │
     ┌─────────────────────────────┐                   │ 4. Grounded Answer
     │   ChromaDB Cloud (HNSW)     │                   ▼
     │  - 3.4 Lakh Indexed Chunks  │    ┌─────────────────────────────┐
     │  - Canonical Source URLs    │───>│   Verifiable Citations JSON  │
     └─────────────────────────────┘    └─────────────────────────────┘
```

- **Vector Database**: ChromaDB Cloud housing **3.4 Lakh GeeksforGeeks chunks** with HNSW indexing.
- **Embeddings**: Chroma native `DefaultEmbeddingFunction` via C++ ONNX runtime (`all-MiniLM-L6-v2`, 384 dimensions, zero PyTorch footprint, strictly **~105 MB peak RSS memory**).
- **Backend API**: FastAPI with Pydantic v2 schemas, strict type bounds (`top_k: ge=1, le=10`), `@field_validator` against empty queries, and AnyIO worker threadpool offloading.
- **LLM Circuit Breaker**: Dual-provider resilient generation engine (Primary: Groq LPU `openai/gpt-oss-120b` for 2s latency; Fallback: Google Gemini `gemini-3.8-flash` on 429/503 spikes).
- **Frontend UI**: Streamlit modern chat interface with interactive GFG source citation cards, onboarding suggestion pills, and real-time latency badges.
- **Containerization**: Multi-stage `Dockerfile` (Python 3.12-slim, non-root `appuser`, dynamic `$PORT` binding) deployed on Render Free Tier (<512 MB RAM).

---

## 📁 Repository Structure

```text
gfg-rag-assistant/
├── .dockerignore                 # Excludes .git, .env, and local caches from Docker image
├── .env.example                  # Documented environment variables template
├── .gitignore                    # Strict rules ignoring secrets, virtualenvs, and pycache
├── Dockerfile                    # Production lightweight Docker container recipe (<300MB)
├── README.md                     # Portfolio documentation & architecture specification
├── render.yaml                   # Infrastructure-as-Code (IaC) Render Cloud blueprint
├── requirements.txt              # Pinned production dependencies (Strictly NO PyTorch)
│
├── notebooks/                    # Interactive Colab exploration & RAG pipeline
│   └── GeeksforGeeks_Advanced_RAG_Pipeline.ipynb
│
├── scripts/                      # Operational verification & smoke test scripts
│   ├── benchmark_memory.py       # Live RAM profiler (verified at 105.55 MB peak RSS)
│   ├── test_api.py               # Automated HTTP integration & negative test suite
│   ├── test_cli.py               # Terminal CLI end-to-end smoke test
│   └── verify_embeddings.py      # ONNX 384-dimension vector verification
│
└── src/                          # Production Decoupled 3-Tier Source Code
    ├── __init__.py
    ├── main.py                   # Application Gateway (Lifespan pre-warm, CORS, Uvicorn)
    │
    ├── api/                      # Layer 3: Web Presentation & Controller
    │   ├── __init__.py
    │   ├── routes.py             # Route definitions (GET /health, POST /api/v1/query)
    │   └── schemas.py            # Pydantic v2 Request & Response DTO contracts
    │
    ├── core/                     # Layer 1: Core Configuration & App Invariants
    │   ├── __init__.py
    │   └── config.py             # Pydantic BaseSettings, @lru_cache singleton, Fail-Fast
    │
    ├── frontend/                 # Presentation Layer: Streamlit Conversational UI
    │   ├── __init__.py
    │   └── app.py                # Streamlit chat interface with citation expanders
    │
    └── services/                 # Layer 2: Business Logic & External Integrations
        ├── __init__.py
        ├── embedding.py          # ONNX-backed embedding wrapper (384-d)
        ├── generator.py          # Dual-LLM Circuit Breaker (Groq primary + Gemini failover)
        └── retriever.py          # Chroma Cloud semantic search & context formatter
```

---

## 🚦 Milestone Progress

- [x] **Phase 0: Security & Memory Guardrails**
  - Scrubbed raw credentials & committed strict `.gitignore`
  - ONNX embedding engine verified without heavy PyTorch
  - Real RAM benchmark verified at **105.55 MB** via `psutil` (Safely within 512MB RAM cap)
- [x] **Phase 1: Modular Core Architecture**
  - Fail-fast configuration with Pydantic v2 `BaseSettings` & `@lru_cache`
  - Dedicated memory-safe `EmbeddingService` (384-dim normalized floats)
  - Chroma Cloud `RetrieverService` querying 3.4 Lakh GFG chunks with canonical URLs
  - Dual-LLM `GeneratorService` (Groq primary + Gemini failover, `temperature=0.0`)
  - End-to-end CLI smoke test verified (`python scripts/test_cli.py`)
- [x] **Phase 2: Lean FastAPI REST Backend**
  - Pydantic v2 contracts (`QueryRequest`, `QueryResponse`, `Citation`, `HealthResponse`)
  - Endpoints: `GET /health` (liveness), `GET /` (root discovery), `POST /api/v1/query`
  - Pre-warming lifespan context manager eliminating cold-start delay
  - Safe CORS middleware supporting configurable origins
  - Automated integration test suite (`scripts/test_api.py`) verifying positive & 422 negative tests
- [x] **Phase 3: Dockerization & Cloud Deployment**
  - Multi-stage production `Dockerfile` with Python 3.12-slim and non-root `appuser`
  - Infrastructure-as-Code `render.yaml` specification with encrypted secret injection
  - **Live Cloud Deployment verified on Render:** [https://gfg-rag-assistant.onrender.com/docs](https://gfg-rag-assistant.onrender.com/docs)
- [x] **Phase 4: Streamlit Chat Frontend**
  - Interactive conversational chat interface in `src/frontend/app.py`
  - Verifiable citation cards displaying GFG article links & similarity match percentage
  - Real-time latency badges, provider telemetry, and interactive suggestion chips

---

## 🛠️ Quickstart (Run Locally)

### 1. Clone & Setup
```bash
git clone https://github.com/sahilgoury7/gfg-rag-assistant.git
cd gfg-rag-assistant
cp .env.example .env
# Fill in your CHROMA_API_KEY, GROQ_API_KEY, and GEMINI_API_KEY in .env
```

### 2. Run FastAPI Backend
```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
# Interactive API Docs at: http://127.0.0.1:8000/docs
```

### 3. Run Streamlit Frontend
```bash
python -m streamlit run src/frontend/app.py
# Chat UI live at: http://localhost:8501
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
