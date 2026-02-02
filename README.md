# 🏭 Agentic RAG System for Manufacturing

An intelligent document Q&A system built for manufacturing environments. Upload equipment manuals, safety procedures, and maintenance logs, then ask questions in natural language.

## ✨ Features

- **Multi-format Document Ingestion**: PDF, DOCX, Excel, PowerPoint, TXT
- **Agentic Query Processing**: Router → Retriever → Generator pipeline
- **Smart Query Decomposition**: Breaks complex questions into sub-queries
- **Source Citations**: Every answer includes document references with page numbers
- **Local-First**: Runs completely offline with Ollama + Milvus Lite

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Vector DB | Milvus Lite (file-based, no Docker needed) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (qwen3:8b) / Gemini API |
| Backend | FastAPI |
| UI | Streamlit |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Ollama (optional, for local LLM)

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to set your LLM provider and API keys
```

### 3. (Optional) Start Ollama for Local LLM

```bash
ollama pull qwen3:8b
ollama serve
```

### 4. Run the Application

**Option A: Streamlit UI**
```bash
streamlit run app.py
```

**Option B: FastAPI Backend**
```bash
uvicorn api:app --reload
```

The app uses **Milvus Lite** which automatically creates a local `milvus_data.db` file - no Docker or separate database setup needed!

## 📁 Project Structure

```
agentic-rag/
├── app.py              # Streamlit UI
├── api.py              # FastAPI backend
├── config.py           # Configuration
├── agents/
│   ├── router.py       # Intent classification
│   ├── retriever.py    # Document search
│   ├── generator.py    # Response generation
│   └── orchestrator.py # Agent coordination
├── ingestion/
│   ├── loader.py       # Document parsing
│   └── chunker.py      # Text chunking
├── vectordb/
│   └── milvus_client.py # Milvus Lite operations
└── data/
    └── samples/        # Sample manufacturing docs
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Ask a question |
| POST | `/ingest` | Upload a document |
| GET | `/documents` | List uploaded documents |
| GET | `/stats` | Get collection stats |
| DELETE | `/reset` | Reset the database |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the maintenance procedure for Pump A?"}'
```

## 🎯 Key Design Decisions

### 1. Agentic Architecture
The system uses a three-agent pipeline:
- **Router**: Classifies intent (retrieval/direct/multi-part)
- **Retriever**: Searches documents with semantic similarity
- **Generator**: Creates responses with mandatory citations

### 2. Manufacturing-Focused
- Sample data includes equipment inventories, maintenance manuals, and safety SOPs
- Citations include page numbers for audit trails
- Supports terminology like LOTO, PPE, and equipment IDs

### 3. On-Premise Capable
- Ollama integration for air-gapped deployments
- Milvus Lite runs as a local file (no Docker needed)
- All processing happens on-device

## 📊 Sample Queries

Try these with the sample data:

1. "What is the LOTO procedure for maintenance?"
2. "What equipment is in Building 1?"
3. "What are the pressure settings for Pump A?"
4. "What PPE is required for hydraulic maintenance?"

## 🔒 Security Considerations

- Documents stay on your infrastructure
- No data sent to external services (when using Ollama)
- Suitable for sensitive manufacturing documentation

## 📝 License

MIT License
