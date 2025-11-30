#!/bin/bash
# Stop backend server

echo "Stopping backend server..."

# Kill processes running main.py
pkill -f "python.*main.py"
pkill -f "python.*back.main"

# Kill uvicorn processes
pkill -f uvicorn

# Kill processes using port 8000
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti :8000)
    if [ ! -z "$PID" ]; then
        echo "Killing process on port 8000: $PID"
        kill -9 $PID 2>/dev/null
    fi
fi

# Kill processes using port 8001 (WebSocket server)
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti :8001)
    if [ ! -z "$PID" ]; then
        echo "Killing process on port 8001: $PID"
        kill -9 $PID 2>/dev/null
    fi
fi

sleep 1

# Verify
if lsof -i :8000 &> /dev/null || lsof -i :8001 &> /dev/null; then
    echo "Warning: Ports 8000 or 8001 are still in use."
    lsof -i :8000 2>/dev/null
    lsof -i :8001 2>/dev/null
else
    echo "Backend server stopped."
fi
