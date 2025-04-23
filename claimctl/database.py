"""Database models and utilities for claim-assistant."""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import get_config

# Initialize SQLAlchemy
Base = declarative_base()


class PageChunk(Base):
    """Model representing a chunk of text from a PDF page."""
    
    __tablename__ = "page_chunks"
    
    id = sa.Column(sa.Integer, primary_key=True)
    file_path = sa.Column(sa.String, nullable=False)
    file_name = sa.Column(sa.String, nullable=False)
    page_num = sa.Column(sa.Integer, nullable=False)
    page_hash = sa.Column(sa.String, nullable=False, index=True, unique=True)
    image_path = sa.Column(sa.String, nullable=False)
    text = sa.Column(sa.Text, nullable=False)
    chunk_type = sa.Column(sa.String, nullable=True)
    
    # Metadata fields
    doc_date = sa.Column(sa.Date, nullable=True)
    doc_id = sa.Column(sa.String, nullable=True)
    processed_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    
    # Add index for common queries
    __table_args__ = (
        sa.Index('idx_file_page', 'file_name', 'page_num'),
    )


def get_database_engine() -> sa.engine.Engine:
    """Get SQLAlchemy engine for the database."""
    config = get_config()
    db_path = Path(config.paths.INDEX_DIR) / "catalog.db"
    db_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Create SQLite engine
    engine = sa.create_engine(f"sqlite:///{db_path}")
    
    return engine


def init_database() -> None:
    """Initialize the database schema."""
    engine = get_database_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    engine = get_database_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def is_page_processed(page_hash: str) -> bool:
    """Check if a page has already been processed."""
    with get_session() as session:
        return session.query(sa.exists().where(PageChunk.page_hash == page_hash)).scalar()


def save_page_chunk(chunk_data: Dict[str, Any]) -> None:
    """Save a page chunk to the database."""
    with get_session() as session:
        chunk = PageChunk(**chunk_data)
        session.add(chunk)
        session.commit()


def get_total_pages() -> int:
    """Get the total number of pages in the database."""
    with get_session() as session:
        return session.query(PageChunk).count()


def get_top_chunks_by_similarity(
    vector_ids: List[int], top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get the top chunks by similarity score."""
    from .utils import console
    
    if not top_k:
        config = get_config()
        top_k = config.retrieval.TOP_K
    
    # Handle empty vector_ids case
    if not vector_ids:
        console.log(f"Looking up chunks for vector_ids: {vector_ids}")
        console.log("No vector IDs provided, falling back to get all chunks")
        with get_session() as session:
            total_chunks = session.query(PageChunk).count()
            console.log(f"Total chunks in database: {total_chunks}")
            
            if total_chunks > 0:
                # Get the most recent chunks up to top_k
                chunks = []
                db_chunks = session.query(PageChunk).order_by(PageChunk.id.desc()).limit(top_k).all()
                
                for chunk in db_chunks:
                    console.log(f"Found chunk: {chunk.file_name} (page {chunk.page_num})")
                    chunks.append({
                        "id": chunk.id,
                        "file_name": chunk.file_name,
                        "file_path": chunk.file_path,
                        "page_num": chunk.page_num,
                        "image_path": chunk.image_path,
                        "text": chunk.text,
                        "chunk_type": chunk.chunk_type,
                        "doc_date": chunk.doc_date.isoformat() if chunk.doc_date else None,
                        "doc_id": chunk.doc_id,
                    })
                
                console.log(f"Returning {len(chunks)} chunks")
                return chunks
            else:
                console.log("No chunks in database")
                return []
    
    # Normal processing for non-empty vector_ids
    console.log(f"Looking up chunks for vector_ids: {vector_ids[:top_k]}")
    
    with get_session() as session:
        # Count total chunks in database
        total_chunks = session.query(PageChunk).count()
        console.log(f"Total chunks in database: {total_chunks}")
        
        chunks = []
        for idx in vector_ids[:top_k]:
            # FAISS vector IDs are 0-indexed, but database IDs are 1-indexed
            chunk = session.query(PageChunk).filter(PageChunk.id == idx + 1).first()
            
            if not chunk and total_chunks > 0:
                console.log(f"Chunk with ID {idx + 1} not found, falling back to more reliable lookup")
                # Try to find the chunk by position in the database
                if 0 <= idx < total_chunks:
                    chunk = session.query(PageChunk).order_by(PageChunk.id).offset(idx).limit(1).first()
            
            if chunk:
                console.log(f"Found chunk: {chunk.file_name} (page {chunk.page_num})")
                chunks.append({
                    "id": chunk.id,
                    "file_name": chunk.file_name,
                    "file_path": chunk.file_path,
                    "page_num": chunk.page_num,
                    "image_path": chunk.image_path,
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    "doc_date": chunk.doc_date.isoformat() if chunk.doc_date else None,
                    "doc_id": chunk.doc_id,
                })
            else:
                console.log(f"No chunk found for idx {idx}")
        
        console.log(f"Returning {len(chunks)} chunks")
        return chunks
