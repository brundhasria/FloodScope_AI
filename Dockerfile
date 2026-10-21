# FloodScope AI - Production Docker Container
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies needed for rasterio / geospatial wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and processed sample artifacts
COPY src/ /app/src/
COPY dataset/processed/ /app/dataset/processed/
COPY scripts/ /app/scripts/

# Ensure output directories exist
RUN mkdir -p /app/dataset/processed/runs

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI server
CMD ["sh", "-c", "uvicorn floodscope.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
