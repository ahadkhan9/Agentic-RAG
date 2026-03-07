"""
FastAPI Backend

REST API for document ingestion, querying, and SSE streaming.
Accepts per-request Gemini API key via X-API-Key header.
"""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.orchestrator import process_query, process_query_stream, ingest_document
from vectordb.milvus_client import get_milvus_client
from vectordb.doc_registry import list_documents as list_registered_docs
from utils import validate_api_key
from config import config

app = FastAPI(
    title="Agentic RAG API",
    description="Document Q&A System with SSE streaming",
    version="3.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = config.max_upload_size_mb * 1024 * 1024


# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------

def get_api_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key", "").strip()
    if not api_key:
        api_key = config.google_api_key
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key format.")
    return api_key


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    intent: str
    response: str
    citations: list[dict]
    sub_queries: Optional[list[str]] = None


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    summary: str
    topics: str
    message: str


class StatsResponse(BaseModel):
    collection_exists: bool
    collection_name: str
    num_documents: int


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Agentic RAG API", "version": "3.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, api_key: str = Depends(get_api_key)):
    try:
        result = process_query(request.query, top_k=request.top_k, api_key=api_key)
        return QueryResponse(
            query=result.query, intent=result.intent.value,
            response=result.response,
            citations=[
                {"source_file": c.source_file, "page_number": c.page_number,
                 "section": c.section, "excerpt": c.excerpt}
                for c in result.citations
            ],
            sub_queries=result.sub_queries,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal processing error.")


# --------------------------------------------------------------------------
# SSE Streaming via StreamingResponse
# --------------------------------------------------------------------------

@app.post("/query/stream")
def query_stream(request: QueryRequest, api_key: str = Depends(get_api_key)):
    """Stream query response via SSE.

    Events:
    - event: meta   — JSON with intent, citations, sub_queries
    - event: token  — text chunk
    - event: done   — end of stream
    """
    stream_ctx, text_stream = process_query_stream(
        request.query, top_k=request.top_k, api_key=api_key,
    )

    def sse_generator():
        # Meta event
        meta = {
            "intent": stream_ctx.intent.value,
            "sub_queries": stream_ctx.sub_queries or [],
            "citations": [
                {"source_file": c.source_file, "page_number": c.page_number,
                 "section": c.section, "excerpt": c.excerpt}
                for c in stream_ctx.citations
            ],
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Token events
        for chunk in text_stream:
            # Escape newlines in SSE data
            escaped = chunk.replace("\n", "\ndata: ")
            yield f"event: token\ndata: {escaped}\n\n"

        # Done event
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    allowed_extensions = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported: {file_ext}")

    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Max: {config.max_upload_size_mb} MB")

    file_path = UPLOAD_DIR / safe_filename
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file.")

    try:
        stats = ingest_document(str(file_path), api_key=api_key)
        return IngestResponse(
            filename=safe_filename, chunks_created=stats["chunks_inserted"],
            summary=stats.get("summary", ""), topics=stats.get("topics", ""),
            message=f"Ingested {safe_filename}",
        )
    except Exception:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to ingest.")


# --------------------------------------------------------------------------
# Documents & Stats
# --------------------------------------------------------------------------

@app.get("/documents")
async def list_documents():
    docs = list_registered_docs()
    return {"documents": [
        {"doc_id": d.doc_id, "filename": d.filename, "summary": d.summary,
         "topics": d.topics, "chunk_count": d.chunk_count, "total_chars": d.total_chars}
        for d in docs
    ]}


@app.get("/stats", response_model=StatsResponse)
async def get_stats(api_key: str = Depends(get_api_key)):
    try:
        client = get_milvus_client(api_key=api_key)
        stats = client.get_collection_stats()
        return StatsResponse(
            collection_exists=stats.get("exists", False),
            collection_name=stats.get("name", ""),
            num_documents=stats.get("num_entities", 0),
        )
    except Exception:
        return StatsResponse(collection_exists=False, collection_name="", num_documents=0)


@app.delete("/reset")
async def reset_collection(api_key: str = Depends(get_api_key)):
    try:
        client = get_milvus_client(api_key=api_key)
        client.delete_collection()
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.name != ".gitkeep":
                f.unlink()
        return {"message": "Reset successfully"}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to reset.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
