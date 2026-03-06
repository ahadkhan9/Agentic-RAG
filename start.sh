#!/bin/bash
set -e

echo "Starting Agentic RAG services..."

# Start FastAPI in background with limited workers for 2GB RAM
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 10 &

# Start Streamlit in foreground
exec streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.maxUploadSize 20 \
    --browser.gatherUsageStats false
