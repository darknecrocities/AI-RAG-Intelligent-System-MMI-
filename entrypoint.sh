#!/bin/bash

# Start the FastAPI application on port 8000
echo "[+] Starting FastAPI Application..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
