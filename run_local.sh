#!/bin/bash

# MMI-RAG CLI Launcher & Server Control script
echo "========================================================="
echo "       MMI-KNOWLEDGE RAG SYSTEM LOCAL LAUNCHER"
echo "========================================================="

# 1. Free up Port 8000 if occupied
PORT=8000
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
    echo "[!] Port $PORT is already in use by process $PID."
    echo "[+] Terminating existing process $PID to clear the port..."
    kill -9 $PID
    sleep 1
    echo "✓ Port $PORT is now free."
else
    echo "✓ Port $PORT is clear."
fi

# 2. Check local Ollama status
echo "[+] Checking local Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✓ Ollama daemon is running."
else
    echo "✗ Ollama is NOT running!"
    echo "  Please start the Ollama application on your Mac first, then re-run this script."
    exit 1
fi

# 3. Detect available models and ensure primary model exists
echo "[+] Detecting best available model..."
EXISTING_MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)

if echo "$EXISTING_MODELS" | grep -q "llama3.2"; then
    echo "✓ Model 'llama3.2' found."
    export MODEL_NAME="llama3.2"
elif echo "$EXISTING_MODELS" | grep -q "qwen2.5"; then
    AVAILABLE=$(echo "$EXISTING_MODELS" | grep "qwen2.5" | head -1)
    echo "⚠️  llama3.2 not ready yet. Falling back to: ${AVAILABLE}"
    export MODEL_NAME="${AVAILABLE}"
else
    echo "[+] No suitable model found. Pulling llama3.2 now (this may take a while)..."
    ollama pull llama3.2
    export MODEL_NAME="llama3.2"
fi
echo "[+] Using model: $MODEL_NAME"

# Pull llama3.2 in background if we're currently using a fallback
if [ "$MODEL_NAME" != "llama3.2" ]; then
    echo "[+] Downloading llama3.2 in background..."
    echo "[+] Monitor: tail -f ollama_pull.log"
    (
        until ollama pull llama3.2 >> ollama_pull.log 2>&1; do
            echo "[!] Pull interrupted. Retrying in 5 seconds..." >> ollama_pull.log
            sleep 5
        done
        echo "✓ llama3.2 is ready! Restart the server to use it." >> ollama_pull.log
    ) &
fi

# 4. Activate Virtual Environment and Launch Uvicorn
if [ -d "venv" ]; then
    echo "[+] Activating Python virtual environment..."
    source venv/bin/activate
else
    echo "✗ Python virtual environment 'venv' not found. Please run installation steps first."
    exit 1
fi

echo "[+] Starting FastAPI server (model: ${MODEL_NAME})..."
echo "👉 Open your browser at http://127.0.0.1:8000/"
echo "---------------------------------------------------------"
exec env OLLAMA_MODEL="${MODEL_NAME}" uvicorn api:app --host 127.0.0.1 --port 8000
