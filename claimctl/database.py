"""Database models and utilities for claim-assistant."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.sql import func

from .config import get_config

# Initialize SQLAlchemy
Base = declarative_base()


class Matter(Base):
    """Model representing a legal matter."""
    __tablename__ = "matters"

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False, unique=True)
    description = sa.Column(sa.String, nullable=True)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    last_accessed = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_directory = sa.Column(sa.String, nullable=False)
    index_directory = sa.Column(sa.String, nullable=False)
    settings = sa.Column(sa.JSON, nullable=True)  # Matter-specific settings

    # Define relationship with documents
    documents = relationship("Document", back_populates="matter")


class Document(Base):
    """Model representing a document."""

    __tablename__ = "documents"

    id = sa.Column(sa.Integer, primary_key=True)
    file_path = sa.Column(sa.String, nullable=False, unique=True)
    file_name = sa.Column(sa.String, nullable=False)
    # New metadata fields
    project_name = sa.Column(sa.String, nullable=True)
    document_date = sa.Column(sa.Date, nullable=True)
    document_type = sa.Column(sa.String, nullable=True)
    document_id = sa.Column(sa.String, nullable=True)
    parties_involved = sa.Column(sa.String, nullable=True)
    # Versioning fields
    version = sa.Column(sa.Integer, default=1, nullable=False)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(
        sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    # Matter relationship
    matter_id = sa.Column(sa.Integer, sa.ForeignKey("matters.id"), nullable=False)
    matter = relationship("Matter", back_populates="documents")

    # Define relationship with pages
    pages = relationship(
        "Page", back_populates="document", cascade="all, delete-orphan"
    )


class Page(Base):
    """Model representing a page of a document."""

    __tablename__ = "pages"

    id = sa.Column(sa.Integer, primary_key=True)
    document_id = sa.Column(sa.Integer, sa.ForeignKey("documents.id"), nullable=False)
    page_num = sa.Column(sa.Integer, nullable=False)
    page_hash = sa.Column(sa.String, nullable=False, index=True, unique=True)
    image_path = sa.Column(sa.String, nullable=False)
    thumbnail_path = sa.Column(sa.String, nullable=True)  # Path to thumbnail image
    processed_at = sa.Column(sa.DateTime, default=datetime.utcnow)

    # Define relationship with document
    document = relationship("Document", back_populates="pages")
    # Define relationship with chunks
    chunks = relationship(
        "PageChunk", back_populates="page", cascade="all, delete-orphan"
    )

    # Ensure page numbers are unique within a document
    __table_args__ = (
        sa.UniqueConstraint("document_id", "page_num", name="uix_document_page"),
    )


class PageChunk(Base):
    """Model representing a chunk of text from a PDF page."""

    __tablename__ = "page_chunks"

    id = sa.Column(sa.Integer, primary_key=True)
    page_id = sa.Column(sa.Integer, sa.ForeignKey("pages.id"), nullable=False)
    chunk_index = sa.Column(sa.Integer, nullable=True)
    total_chunks = sa.Column(sa.Integer, nullable=True)
    text = sa.Column(sa.Text, nullable=False)
    chunk_type = sa.Column(sa.String, nullable=True)
    confidence = sa.Column(sa.Integer, nullable=True)

    # Add FAISS vector ID directly (critical for ensuring proper ID mapping)
    faiss_id = sa.Column(sa.Integer, nullable=True, index=True)

    # Document metadata fields that can be different per chunk
    doc_date = sa.Column(sa.Date, nullable=True)
    doc_id = sa.Column(sa.String, nullable=True)
    
    # Project name can also be stored at chunk level
    project_name = sa.Column(sa.String, nullable=True)

    # Add a unique chunk_id
    chunk_id = sa.Column(sa.String, nullable=True, index=True, unique=True)

    # Add timestamp for indexing/sorting
    processed_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    
    # Add new metadata fields
    amount = sa.Column(sa.String, nullable=True)
    time_period = sa.Column(sa.String, nullable=True)
    section_reference = sa.Column(sa.String, nullable=True)
    public_agency_reference = sa.Column(sa.String, nullable=True)
    work_description = sa.Column(sa.String, nullable=True)

    # Define relationship with page
    page = relationship("Page", back_populates="chunks")

    # Add indices for common queries
    __table_args__ = (
        sa.Index("idx_chunk_type", "chunk_type"),
        sa.Index("idx_doc_date", "doc_date"),
        sa.Index("idx_faiss_id", "faiss_id"),
        sa.Index("idx_project_name", "project_name"),
        sa.Index("idx_section_reference", "section_reference"),
        sa.Index("idx_public_agency_reference", "public_agency_reference"),
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
    """Check if a page has already been processed.

    Now that pages can have multiple chunks, we just check if any chunks exist
    for this page hash, as an indicator of whether this page has been processed.
    """
    with get_session() as session:
        page = session.query(Page).filter(Page.page_hash == page_hash).first()
        return page is not None


def save_page_chunk(chunk_data: Dict[str, Any]) -> None:
    """Save a page chunk to the database."""
    with get_session() as session:
        try:
            # Start transaction
            transaction = session.begin_nested()

            # Extract document and page data from chunk data
            file_path = chunk_data.pop("file_path")
            file_name = chunk_data.pop("file_name")
            page_num = chunk_data.pop("page_num")
            page_hash = chunk_data.pop("page_hash")
            image_path = chunk_data.pop("image_path")
            thumbnail_path = chunk_data.pop(
                "thumbnail_path", None
            )  # May not exist in older data

            # Extra metadata that can be applied to the document
            project_name = chunk_data.pop("project_name", None)
            document_type = chunk_data.get("chunk_type", None)
            document_id = chunk_data.get("doc_id", None)
            doc_date = chunk_data.get("doc_date", None)

            # Extract matter_id if available
            matter_id = chunk_data.pop("matter_id", None)
            if not matter_id:
                raise ValueError("matter_id is required for document creation")
            
            # Check if document exists, create if not
            document = (
                session.query(Document).filter(Document.file_path == file_path).first()
            )

            if not document:
                # Get parties_involved if available
                parties_involved = chunk_data.pop("parties_involved", None)

                document = Document(
                    file_path=file_path,
                    file_name=file_name,
                    project_name=project_name,
                    document_type=document_type,
                    document_id=document_id,
                    document_date=doc_date,
                    parties_involved=parties_involved,
                    matter_id=matter_id,  # Add matter_id
                )
                session.add(document)
                session.flush()  # Ensure ID is assigned

            # Check if page exists, create if not
            page = session.query(Page).filter(Page.page_hash == page_hash).first()

            if not page:
                page = Page(
                    document_id=document.id,
                    page_num=page_num,
                    page_hash=page_hash,
                    image_path=image_path,
                    thumbnail_path=thumbnail_path,
                )
                session.add(page)
                session.flush()  # Ensure ID is assigned

            # Create the page chunk
            chunk = PageChunk(page_id=page.id, **chunk_data)
            session.add(chunk)

            # Commit transaction
            transaction.commit()

        except Exception as e:
            # Rollback transaction on error
            if transaction.is_active:
                transaction.rollback()
            raise e

        # Commit the session
        session.commit()


def save_faiss_id_mapping(chunk_id: str, faiss_id: int) -> None:
    """Save the mapping between chunk_id and FAISS vector ID."""
    with get_session() as session:
        try:
            # Find the chunk by chunk_id
            chunk = (
                session.query(PageChunk).filter(PageChunk.chunk_id == chunk_id).first()
            )

            if chunk:
                # Update the FAISS ID
                chunk.faiss_id = faiss_id
                session.commit()
        except Exception as e:
            session.rollback()
            raise e


def get_document_by_path(file_path: str) -> Optional[Dict[str, Any]]:
    """Get a document by file path."""
    with get_session() as session:
        document = (
            session.query(Document).filter(Document.file_path == file_path).first()
        )

        if document:
            return {
                "id": document.id,
                "file_path": document.file_path,
                "file_name": document.file_name,
                "project_name": document.project_name,
                "document_type": document.document_type,
                "document_id": document.document_id,
                "document_date": document.document_date,
                "parties_involved": document.parties_involved,
                "version": document.version,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
            }

        return None


def update_document_metadata(file_path: str, metadata: Dict[str, Any]) -> None:
    """Update document metadata."""
    with get_session() as session:
        document = (
            session.query(Document).filter(Document.file_path == file_path).first()
        )

        if document:
            for key, value in metadata.items():
                if hasattr(document, key):
                    setattr(document, key, value)

            # Increment version
            document.version += 1

            session.commit()


def get_chunk_by_faiss_id(faiss_id: int) -> Optional[Dict[str, Any]]:
    """Get a chunk by FAISS ID."""
    with get_session() as session:
        chunk = session.query(PageChunk).filter(PageChunk.faiss_id == faiss_id).first()

        if chunk:
            # Join with page and document
            page = session.query(Page).filter(Page.id == chunk.page_id).first()
            document = (
                session.query(Document).filter(Document.id == page.document_id).first()
            )

            return {
                "id": chunk.id,
                "faiss_id": chunk.faiss_id,
                "file_name": document.file_name,
                "file_path": document.file_path,
                "page_num": page.page_num,
                "image_path": page.image_path,
                "thumbnail_path": page.thumbnail_path,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "confidence": chunk.confidence,
                "doc_date": chunk.doc_date.isoformat() if chunk.doc_date else None,
                "doc_id": chunk.doc_id,
                "project_name": document.project_name,
                "document_type": document.document_type,
                "parties_involved": document.parties_involved,
            }

        return None


def get_total_pages() -> int:
    """Get the total number of pages in the database."""
    with get_session() as session:
        return session.query(Page).count()


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
                chunks = []
                # Get the most recent chunks up to top_k
                db_chunks = (
                    session.query(PageChunk)
                    .order_by(PageChunk.processed_at.desc())
                    .limit(top_k)
                    .all()
                )

                for chunk in db_chunks:
                    # Get page and document information
                    page = session.query(Page).filter(Page.id == chunk.page_id).first()
                    if not page:
                        continue

                    document = (
                        session.query(Document)
                        .filter(Document.id == page.document_id)
                        .first()
                    )
                    if not document:
                        continue

                    console.log(
                        f"Found chunk: {document.file_name} (page {page.page_num})"
                    )
                    chunks.append(
                        {
                            "id": chunk.id,
                            "faiss_id": chunk.faiss_id,
                            "file_name": document.file_name,
                            "file_path": document.file_path,
                            "page_num": page.page_num,
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "image_path": page.image_path,
                            "thumbnail_path": page.thumbnail_path,
                            "text": chunk.text,
                            "chunk_type": chunk.chunk_type,
                            "confidence": chunk.confidence,
                            "doc_date": (
                                chunk.doc_date.isoformat() if chunk.doc_date else None
                            ),
                            "doc_id": chunk.doc_id,
                            "project_name": document.project_name,
                            "document_type": document.document_type,
                            "parties_involved": document.parties_involved,
                        }
                    )

                console.log(f"Returning {len(chunks)} chunks")
                return chunks
            else:
                console.log("No chunks in database")
                return []

    # Normal processing for non-empty vector_ids
    console.log(f"Looking up chunks for vector_ids: {vector_ids[:top_k]}")

    with get_session() as session:
        chunks = []
        for faiss_id in vector_ids[:top_k]:
            # Look up by faiss_id directly
            chunk = (
                session.query(PageChunk).filter(PageChunk.faiss_id == faiss_id).first()
            )

            if chunk:
                # Get page and document information
                page = session.query(Page).filter(Page.id == chunk.page_id).first()
                if not page:
                    continue

                document = (
                    session.query(Document)
                    .filter(Document.id == page.document_id)
                    .first()
                )
                if not document:
                    continue

                console.log(f"Found chunk: {document.file_name} (page {page.page_num})")
                chunks.append(
                    {
                        "id": chunk.id,
                        "faiss_id": chunk.faiss_id,
                        "file_name": document.file_name,
                        "file_path": document.file_path,
                        "page_num": page.page_num,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "image_path": page.image_path,
                        "thumbnail_path": page.thumbnail_path,
                        "text": chunk.text,
                        "chunk_type": chunk.chunk_type,
                        "confidence": chunk.confidence,
                        "doc_date": (
                            chunk.doc_date.isoformat() if chunk.doc_date else None
                        ),
                        "doc_id": chunk.doc_id,
                        "project_name": document.project_name,
                        "document_type": document.document_type,
                        "parties_involved": document.parties_involved,
                    }
                )
            else:
                console.log(f"No chunk found for FAISS ID {faiss_id}")

        console.log(f"Returning {len(chunks)} chunks")
        return chunks


def get_chunks_by_metadata(
    metadata_filters: Dict[str, Any], limit: int = 10
) -> List[Dict[str, Any]]:
    """Get chunks based on metadata filters."""
    with get_session() as session:
        query = session.query(PageChunk)

        # Join tables for document metadata
        query = query.join(Page, Page.id == PageChunk.page_id)
        query = query.join(Document, Document.id == Page.document_id)

        # Apply filters
        for key, value in metadata_filters.items():
            if key == "document_type" and value:
                query = query.filter(Document.document_type == value)
            elif key == "project_name" and value:
                query = query.filter(Document.project_name == value)
            elif key == "parties_involved" and value:
                query = query.filter(Document.parties_involved.like(f"%{value}%"))
            elif key == "chunk_type" and value:
                query = query.filter(PageChunk.chunk_type == value)
            elif key == "date_from" and value:
                query = query.filter(PageChunk.doc_date >= value)
            elif key == "date_to" and value:
                query = query.filter(PageChunk.doc_date <= value)
            # Add new filters
            elif key == "amount_min" and value:
                # Convert string amounts to numeric for comparison
                query = query.filter(
                    sa.cast(sa.func.replace(sa.func.replace(PageChunk.amount, "$", ""), ",", ""), sa.Float) >= value
                )
            elif key == "amount_max" and value:
                query = query.filter(
                    sa.cast(sa.func.replace(sa.func.replace(PageChunk.amount, "$", ""), ",", ""), sa.Float) <= value
                )
            elif key == "section_reference" and value:
                query = query.filter(PageChunk.section_reference == value)
            elif key == "public_agency" and value:
                query = query.filter(PageChunk.public_agency_reference != None)

        # Get results
        chunks = query.order_by(PageChunk.processed_at.desc()).limit(limit).all()

        result = []
        for chunk in chunks:
            # Get page and document information
            page = session.query(Page).filter(Page.id == chunk.page_id).first()
            if not page:
                continue

            document = (
                session.query(Document).filter(Document.id == page.document_id).first()
            )
            if not document:
                continue

            result.append(
                {
                    "id": chunk.id,
                    "faiss_id": chunk.faiss_id,
                    "file_name": document.file_name,
                    "file_path": document.file_path,
                    "page_num": page.page_num,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "image_path": page.image_path,
                    "thumbnail_path": page.thumbnail_path,
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    "confidence": chunk.confidence,
                    "doc_date": (
                        chunk.doc_date.isoformat() if chunk.doc_date else None
                    ),
                    "doc_id": chunk.doc_id,
                    "project_name": document.project_name,
                    "document_type": document.document_type,
                    "parties_involved": document.parties_involved,
                }
            )

        return result
