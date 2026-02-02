"""
FastAPI Backend

REST API for document ingestion and querying.
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import process_query, ingest_document
from vectordb.milvus_client import get_milvus_client

app = FastAPI(
    title="Agentic RAG API",
    description="Manufacturing Document Q&A System",
    version="1.0.0"
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    """Request model for querying."""
    query: str
    top_k: int = 5


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


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Agentic RAG API"}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document database.
    
    The query is processed through:
    1. Router Agent - classifies intent
    2. Retriever Agent - finds relevant documents
    3. Generator Agent - creates response with citations
    """
    try:
        result = process_query(request.query, top_k=request.top_k)
        
        # Convert citations to dict format
        citations = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "section": c.section,
                "excerpt": c.excerpt
            }
            for c in result.citations
        ]
        
        return QueryResponse(
            query=result.query,
            intent=result.intent.value,
            response=result.response,
            citations=citations,
            sub_queries=result.sub_queries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest a document.
    
    Supported formats: PDF, DOCX, XLSX, PPTX, TXT
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.xlsx', '.xls', '.pptx', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Ingest document
    try:
        stats = ingest_document(str(file_path))
        return IngestResponse(
            filename=file.filename,
            chunks_created=stats["chunks_inserted"],
            message=f"Successfully ingested {file.filename}"
        )
    except Exception as e:
        # Clean up file on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.name != ".gitkeep":
            files.append({
                "filename": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2)
            })
    return {"documents": files}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get collection statistics."""
    try:
        client = get_milvus_client()
        stats = client.get_collection_stats()
        return StatsResponse(
            collection_exists=stats.get("exists", False),
            collection_name=stats.get("name", ""),
            num_documents=stats.get("num_entities", 0)
        )
    except Exception as e:
        return StatsResponse(
            collection_exists=False,
            collection_name="",
            num_documents=0
        )


@app.delete("/reset")
async def reset_collection():
    """Delete all documents and reset the collection."""
    try:
        client = get_milvus_client()
        client.delete_collection()
        
        # Clear upload directory
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.name != ".gitkeep":
                f.unlink()
        
        return {"message": "Collection reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
