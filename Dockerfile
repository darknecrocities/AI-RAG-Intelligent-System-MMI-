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

# Copy lean requirements and install dependencies (no PyTorch!)
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Run entrypoint
ENTRYPOINT ["./entrypoint.sh"]
