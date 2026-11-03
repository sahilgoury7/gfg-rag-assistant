# ==============================================================================
# Production Dockerfile for GFG RAG Assistant
#
# Target Platform: Render Web Service (Free Tier: 512 MB RAM / 0.1 CPU)
# Base Image: Python 3.12-slim (Debian Bookworm, ~150 MB footprint)
# Security: Non-root user execution, no hardcoded secrets, minimal attack surface
# ==============================================================================

FROM python:3.12-slim AS runtime

# Set environment optimizations
# - PYTHONDONTWRITEBYTECODE: Prevents writing .pyc files to save container disk space
# - PYTHONUNBUFFERED: Forces stdout/stderr flush immediately for real-time cloud logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set container working directory
WORKDIR /app

# Install minimal OS dependencies if needed (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Copy ONLY requirements first to leverage Docker layer caching
COPY requirements.txt .

# Step 2: Install dependencies with --no-cache-dir to keep image size under 300 MB
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 3: Copy source code into container
COPY src/ ./src/

# Step 4: Security Hardening — Create non-root system user and grant access
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Liveness Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Uvicorn ASGI server with dynamic PORT resolution (required by Render)
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
