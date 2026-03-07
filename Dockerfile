FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build deps to save space
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create data directories + non-root user
RUN mkdir -p data/uploads data/samples logs && \
    adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["/app/start.sh"]

