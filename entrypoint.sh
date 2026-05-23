#!/bin/bash

# Start Ollama service in the background
echo "[+] Starting Ollama service..."
ollama serve > ollama.log 2>&1 &

# Wait for Ollama to start
echo "[+] Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
        echo "✓ Ollama is ready."
        break
    fi
    echo "    Waiting..."
    sleep 2
done

# Pull the lightweight model configured in config.py
echo "[+] Pulling Ollama model qwen2.5:0.5b..."
ollama pull qwen2.5:0.5b

# Start the FastAPI application on port 8000
echo "[+] Starting FastAPI Application..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
