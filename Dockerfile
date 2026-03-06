FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF, document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/milvus

# Expose ports: FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

# Start script runs both services
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
