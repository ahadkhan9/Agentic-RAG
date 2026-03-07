"""
FastAPI Backend

REST API for document ingestion and querying.
Accepts per-request Gemini API key via X-API-Key header.
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.orchestrator import process_query, ingest_document
from vectordb.milvus_client import get_milvus_client
from utils import validate_api_key
from config import config

app = FastAPI(
    title="Agentic RAG API",
    description="Document Q&A System",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url=None,
)

# CORS — use configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Max upload size in bytes
MAX_UPLOAD_BYTES = config.max_upload_size_mb * 1024 * 1024


# --------------------------------------------------------------------------
# Security: API key extraction
# --------------------------------------------------------------------------

def get_api_key(request: Request) -> str:
    """Extract and validate API key from request header.

    Uses X-API-Key header. Falls back to server-level key if configured.
    """
    api_key = request.headers.get("X-API-Key", "").strip()

    if not api_key:
        api_key = config.google_api_key

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide via X-API-Key header.",
        )

    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format.",
        )

    return api_key


# --------------------------------------------------------------------------
# Request/Response Models
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request model for querying."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Response model for queries."""
    query: str
    intent: str
    response: str
    citations: list[dict]
    sub_queries: Optional[list[str]] = None


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    filename: str
    chunks_created: int
    message: str


class StatsResponse(BaseModel):
    """Response model for collection stats."""
    collection_exists: bool
    collection_name: str
    num_documents: int


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Agentic RAG API", "version": "2.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest, api_key: str = Depends(get_api_key)
):
    """Query the document database.

    Requires X-API-Key header with a valid Gemini API key.
    """
    try:
        result = process_query(
            request.query, top_k=request.top_k, api_key=api_key
        )

        citations = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "section": c.section,
                "excerpt": c.excerpt,
            }
            for c in result.citations
        ]

        return QueryResponse(
            query=result.query,
            intent=result.intent.value,
            response=result.response,
            citations=citations,
            sub_queries=result.sub_queries,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Don't leak internal details
        raise HTTPException(status_code=500, detail="Internal processing error.")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
):
    """Upload and ingest a document.

    Supported formats: PDF, DOCX, XLSX, PPTX, TXT
    Requires X-API-Key header with a valid Gemini API key.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    # Sanitize filename — prevent path traversal
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # Check file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {config.max_upload_size_mb} MB",
        )

    # Save file
    file_path = UPLOAD_DIR / safe_filename
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file.")

    # Ingest document
    try:
        stats = ingest_document(str(file_path), api_key=api_key)
        return IngestResponse(
            filename=safe_filename,
            chunks_created=stats["chunks_inserted"],
            message=f"Successfully ingested {safe_filename}",
        )
    except Exception:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to ingest document.")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.name != ".gitkeep":
            files.append(
                {
                    "filename": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 2),
                }
            )
    return {"documents": files}


@app.get("/stats", response_model=StatsResponse)
async def get_stats(api_key: str = Depends(get_api_key)):
    """Get collection statistics."""
    try:
        client = get_milvus_client(api_key=api_key)
        stats = client.get_collection_stats()
        return StatsResponse(
            collection_exists=stats.get("exists", False),
            collection_name=stats.get("name", ""),
            num_documents=stats.get("num_entities", 0),
        )
    except Exception:
        return StatsResponse(
            collection_exists=False,
            collection_name="",
            num_documents=0,
        )


@app.delete("/reset")
async def reset_collection(api_key: str = Depends(get_api_key)):
    """Delete all documents and reset the collection."""
    try:
        client = get_milvus_client(api_key=api_key)
        client.delete_collection()

        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.name != ".gitkeep":
                f.unlink()

        return {"message": "Collection reset successfully"}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to reset collection.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
