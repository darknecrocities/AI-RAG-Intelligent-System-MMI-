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

# Pull the model configured in config.py (with auto-retry)
echo "[+] Pulling Ollama model llama3.2..."
until ollama pull llama3.2; do
    echo "[!] Pull failed. Retrying..."
    sleep 5
done




# Start the FastAPI application on port 8000
echo "[+] Starting FastAPI Application..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
