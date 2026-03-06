#!/bin/bash
set -e

echo "Starting Agentic RAG services..."

# Start FastAPI in background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
