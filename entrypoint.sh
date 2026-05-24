#!/bin/bash

# Start the FastAPI application with configurable workers (defaults to 1 to prevent OOM on Render Free Tier)
echo "[+] Starting FastAPI Application..."
exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}
