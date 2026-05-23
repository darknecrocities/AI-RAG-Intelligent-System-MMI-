#!/bin/bash

# Start the FastAPI application
echo "[+] Starting FastAPI Application..."
exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
