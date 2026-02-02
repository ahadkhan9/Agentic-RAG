"""
Text Chunking Module (Enhanced)

Splits document content into smaller chunks for embedding.
Features:
- Content-type aware chunking (text, tables, lists)
- Sentence boundary detection
- Configurable overlap for context continuity
- Metadata preservation
"""
from dataclasses import dataclass
from typing import Optional

from config import config
from ingestion.loader import DocumentChunk
from logger import get_logger

logger = get_logger("Chunker")


@dataclass
class TextChunk:
    """A chunk of text ready for embedding with rich metadata."""
    content: str
    source_file: str
    file_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    content_type: str = "text"  # text, table, list, heading


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> list[str]:
    """
    Split text into overlapping chunks with smart boundary detection.
    
    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk (default from config)
        overlap: Number of overlapping characters (default from config)
    
    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or config.chunk_size
    overlap = overlap or config.chunk_overlap
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings (prioritize double newlines, then periods)
            best_break = -1
            for sep in ['\n\n', '. ', '.\n', '! ', '? ', ':\n', '\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size // 2:
                    best_break = start + last_sep + len(sep)
                    break
            
            if best_break > start:
                end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def chunk_table(table_text: str, max_chunk_size: int = None) -> list[str]:
    """
    Smart chunking for table content.
    - Small tables: keep as single chunk
    - Large tables: split by rows, preserving header
    """
    max_chunk_size = max_chunk_size or config.chunk_size * 2  # Tables get more space
    
    if len(table_text) <= max_chunk_size:
        return [table_text]
    
    # Split by lines and identify header
    lines = table_text.strip().split('\n')
    if len(lines) < 3:
        return [table_text]
    
    # Markdown table format: header | separator | rows
    header = lines[0]
    separator = lines[1] if '---' in lines[1] else None
    
    chunks = []
    current_chunk = [header]
    if separator:
        current_chunk.append(separator)
    
    for line in lines[2:]:
        test_chunk = '\n'.join(current_chunk + [line])
        
        if len(test_chunk) > max_chunk_size:
            # Save current chunk and start new one with header
            if len(current_chunk) > 2:  # Has content beyond header
                chunks.append('\n'.join(current_chunk))
            current_chunk = [header]
            if separator:
                current_chunk.append(separator)
        
        current_chunk.append(line)
    
    # Add remaining
    if len(current_chunk) > 2:
        chunks.append('\n'.join(current_chunk))
    
    return chunks if chunks else [table_text]


def chunk_document(doc_chunk: DocumentChunk) -> list[TextChunk]:
    """
    Split a document chunk into smaller text chunks.
    Uses content-type aware chunking strategies.
    """
    content_type = getattr(doc_chunk, 'content_type', 'text')
    
    # Choose chunking strategy based on content type
    if content_type in ('table', 'table_row'):
        text_chunks = chunk_table(doc_chunk.content)
    elif content_type == 'heading':
        # Don't chunk headings
        text_chunks = [doc_chunk.content]
    else:
        text_chunks = chunk_text(doc_chunk.content)
    
    result = []
    for idx, text in enumerate(text_chunks):
        result.append(TextChunk(
            content=text,
            source_file=doc_chunk.source_file,
            file_type=doc_chunk.file_type,
            page_number=doc_chunk.page_number or doc_chunk.slide_number,
            section=doc_chunk.section,
            chunk_index=idx,
            content_type=content_type
        ))
    
    return result


def chunk_documents(doc_chunks: list[DocumentChunk]) -> list[TextChunk]:
    """
    Process multiple document chunks into text chunks.
    Logs processing statistics.
    """
    all_chunks = []
    content_type_counts = {}
    
    for doc_chunk in doc_chunks:
        chunks = chunk_document(doc_chunk)
        all_chunks.extend(chunks)
        
        # Track content types
        content_type = getattr(doc_chunk, 'content_type', 'text')
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
    
    # Log statistics
    logger.info(f"📊 Chunking complete: {len(doc_chunks)} docs → {len(all_chunks)} chunks")
    for ctype, count in content_type_counts.items():
        logger.debug(f"  {ctype}: {count} items")
    
    return all_chunks
