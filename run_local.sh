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

# 3. Pull Llama3.2 model in the background (with auto-retry)
echo "[+] Starting model 'llama3.2' download in the background..."
echo "[+] It will automatically resume if the connection drops."
echo "[+] You can monitor download logs by running: tail -f ollama_pull.log"
(
    until ollama pull llama3.2; do
        echo "[!] Pull failed or interrupted. Retrying in 5 seconds..." >> ollama_pull.log
        sleep 5
    done
    echo "✓ Model llama3.2 is ready." >> ollama_pull.log
) &

# 4. Activate Virtual Environment and Launch Uvicorn
if [ -d "venv" ]; then
    echo "[+] Activating Python virtual environment..."
    source venv/bin/activate
else
    echo "✗ Python virtual environment 'venv' not found. Please run installation steps first."
    exit 1
fi

echo "[+] Starting FastAPI server and mounting Web Dashboard..."
echo "👉 Open your browser at http://127.0.0.1:8000/"
echo "---------------------------------------------------------"
exec uvicorn api:app --host 127.0.0.1 --port 8000 --reload
