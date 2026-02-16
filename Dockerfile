# Backend Dockerfile for NeuralNav
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files and README (hatchling requires README.md for package metadata)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies (frozen = use lockfile exactly, no-dev = skip dev deps)
RUN uv sync --frozen --no-dev

# Copy backend source code
COPY src/neuralnav ./src/neuralnav

# Copy data files (Knowledge Base)
COPY data ./data

# Create directories for generated files and uv cache
RUN mkdir -p /app/generated_configs /app/logs /app/.cache/uv && \
    chmod -R 777 /app/.cache /app/generated_configs /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/app/.cache/uv

ARG MODEL_CATALOG_URL

# Expose backend API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD uv run python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the backend API server
CMD ["uv", "run", "uvicorn", "neuralnav.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
