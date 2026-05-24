FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Enable lightweight mode (uses local CPU PyTorch for embeddings but no CrossEncoder/LLM)
ENV LIGHTWEIGHT_MODE=false

# Copy lean requirements and install dependencies
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Pre-download and cache the embedding model to ensure 100% local, offline embedding execution
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Add Healthcheck to monitor API health
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Run entrypoint
ENTRYPOINT ["./entrypoint.sh"]
