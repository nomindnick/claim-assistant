"""Database models and utilities for claim-assistant."""

import os
from datetime import datetime, date
from enum import Enum
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
    
    # Define relationship with timeline events
    timeline_events = relationship("TimelineEvent", back_populates="matter")


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

    # Define relationship with timeline events
    timeline_events = relationship("TimelineEvent", back_populates="chunk")

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
    
    Also checks for the actual chunks to handle cases where the database was cleared 
    but the resume log wasn't.
    """
    with get_session() as session:
        # First check if the page exists
        page = session.query(Page).filter(Page.page_hash == page_hash).first()
        if not page:
            return False
            
        # Also check if there are any chunks for this page
        # This handles cases where database was partially cleared or migrated
        chunk_count = session.query(PageChunk).filter(PageChunk.page_id == page.id).count()
        return chunk_count > 0


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


def save_timeline_event(event_data: Dict[str, Any]) -> Optional[int]:
    """Save a timeline event to the database.
    
    Args:
        event_data: Dictionary containing timeline event data
        
    Returns:
        ID of the created timeline event, or None if an error occurred
    """
    with get_session() as session:
        try:
            # Start transaction
            transaction = session.begin_nested()
            
            # Create new timeline event
            event = TimelineEvent(**event_data)
            session.add(event)
            session.flush()  # To get the ID
            
            # Commit transaction
            transaction.commit()
            
            # Return the ID of the newly created event
            return event.id
            
        except Exception as e:
            # Rollback transaction on error
            if transaction.is_active:
                transaction.rollback()
            print(f"Error saving timeline event: {str(e)}")
            return None
        
        # Commit the session
        session.commit()


def get_timeline_events(
    matter_id: int,
    event_types: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    document_id: Optional[int] = None,
    min_confidence: float = 0.5,
    min_importance: Optional[float] = None,
    include_contradictions: bool = False,
    include_financial_impacts: bool = False,
    limit: int = 100,
    sort_by: str = "date",
) -> List[Dict[str, Any]]:
    """Get timeline events for a matter, with optional filtering.
    
    Args:
        matter_id: ID of the matter
        event_types: Optional list of event types to filter by
        date_from: Optional start date for filtering
        date_to: Optional end date for filtering
        document_id: Optional document ID to filter by
        min_confidence: Minimum confidence score (0-1)
        min_importance: Minimum importance score
        include_contradictions: Whether to include contradiction details
        include_financial_impacts: Whether to include financial impact details
        limit: Maximum number of events to return
        sort_by: Field to sort by ('date', 'importance', 'confidence', 'financial_impact')
        
    Returns:
        List of timeline events matching the criteria
    """
    with get_session() as session:
        query = session.query(TimelineEvent).filter(TimelineEvent.matter_id == matter_id)
        
        # Apply filters
        if event_types:
            query = query.filter(TimelineEvent.event_type.in_(event_types))
        
        if date_from:
            query = query.filter(TimelineEvent.event_date >= date_from)
        
        if date_to:
            query = query.filter(TimelineEvent.event_date <= date_to)
        
        if document_id:
            query = query.filter(TimelineEvent.document_id == document_id)
        
        if min_confidence:
            query = query.filter(TimelineEvent.confidence >= min_confidence)
        
        if min_importance:
            query = query.filter(TimelineEvent.importance_score >= min_importance)
        
        # Apply sorting
        if sort_by == "date":
            query = query.order_by(TimelineEvent.event_date.asc())
        elif sort_by == "importance":
            query = query.order_by(TimelineEvent.importance_score.desc())
        elif sort_by == "confidence":
            query = query.order_by(TimelineEvent.confidence.desc())
        elif sort_by == "financial_impact":
            query = query.order_by(TimelineEvent.financial_impact.desc())
        else:
            # Default sort by date
            query = query.order_by(TimelineEvent.event_date.asc())
        
        # Apply limit
        events = query.limit(limit).all()
        
        # Format results
        result = []
        for event in events:
            # Get document information
            document = session.query(Document).filter(Document.id == event.document_id).first()
            chunk = session.query(PageChunk).filter(PageChunk.id == event.chunk_id).first()
            
            # Format event data
            event_data = {
                "id": event.id,
                "event_date": event.event_date.isoformat() if event.event_date else None,
                "event_type": event.event_type,
                "description": event.description,
                "importance_score": event.importance_score,
                "confidence": event.confidence,
                "referenced_documents": event.referenced_documents,
                "involved_parties": event.involved_parties,
                "document": {
                    "id": document.id,
                    "file_name": document.file_name,
                    "file_path": document.file_path,
                    "document_type": document.document_type,
                } if document else None,
                "chunk": {
                    "id": chunk.id,
                    "text": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                    "page_num": session.query(Page).filter(Page.id == chunk.page_id).first().page_num if chunk else None,
                } if chunk else None,
            }
            
            # Add financial impact information if present and requested
            if include_financial_impacts:
                event_data["financial_impact"] = event.financial_impact
                event_data["financial_impact_description"] = event.financial_impact_description
                event_data["financial_impact_type"] = event.financial_impact_type
                
                # Get related financial events
                financial_events = session.query(FinancialEvent).filter(
                    FinancialEvent.timeline_event_id == event.id
                ).all()
                
                if financial_events:
                    event_data["financial_events"] = [
                        {
                            "id": fe.id,
                            "amount": fe.amount,
                            "amount_description": fe.amount_description,
                            "currency": fe.currency,
                            "event_type": fe.event_type,
                            "category": fe.category,
                            "event_date": fe.event_date.isoformat() if fe.event_date else None,
                            "is_additive": fe.is_additive,
                            "running_total": fe.running_total,
                        }
                        for fe in financial_events
                    ]
            
            # Add contradiction information if present and requested
            if include_contradictions and event.has_contradiction:
                event_data["has_contradiction"] = True
                event_data["contradiction_details"] = event.contradiction_details
                
                # If there's a contradicting event, include its basic info
                if event.contradicting_event_id:
                    contradicting_event = session.query(TimelineEvent).filter(
                        TimelineEvent.id == event.contradicting_event_id
                    ).first()
                    
                    if contradicting_event:
                        event_data["contradicting_event"] = {
                            "id": contradicting_event.id,
                            "event_date": contradicting_event.event_date.isoformat() if contradicting_event.event_date else None,
                            "event_type": contradicting_event.event_type,
                            "description": contradicting_event.description,
                            "document_id": contradicting_event.document_id,
                        }
            
            result.append(event_data)
        
        return result


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


class TimelineEvent(Base):
    """Model representing a timeline event extracted from documents."""
    
    __tablename__ = "timeline_events"
    
    id = sa.Column(sa.Integer, primary_key=True)
    matter_id = sa.Column(sa.Integer, sa.ForeignKey("matters.id"), nullable=False)
    chunk_id = sa.Column(sa.Integer, sa.ForeignKey("page_chunks.id"), nullable=False)
    document_id = sa.Column(sa.Integer, sa.ForeignKey("documents.id"), nullable=False)
    
    # Event metadata
    event_date = sa.Column(sa.Date, nullable=True, index=True)
    event_type = sa.Column(sa.String, nullable=False, index=True)
    description = sa.Column(sa.Text, nullable=False)
    importance_score = sa.Column(sa.Float, nullable=True)
    confidence = sa.Column(sa.Float, nullable=False)
    
    # Related entities and references
    referenced_documents = sa.Column(sa.String, nullable=True)
    involved_parties = sa.Column(sa.String, nullable=True)
    
    # Contradiction fields
    has_contradiction = sa.Column(sa.Boolean, default=False)
    contradiction_details = sa.Column(sa.Text, nullable=True)
    contradicting_event_id = sa.Column(sa.Integer, nullable=True)
    
    # Financial impact fields
    financial_impact = sa.Column(sa.Float, nullable=True)
    financial_impact_description = sa.Column(sa.String, nullable=True)
    financial_impact_type = sa.Column(sa.String, nullable=True)  # e.g., "cost_increase", "delay_cost", "credit", etc.
    
    # Extraction metadata
    extracted_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    last_updated = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Define relationships
    matter = relationship("Matter", back_populates="timeline_events")
    document = relationship("Document")
    chunk = relationship("PageChunk", back_populates="timeline_events")
    financial_events = relationship("FinancialEvent", back_populates="timeline_event")
    
    # Add indices for querying
    __table_args__ = (
        sa.Index("idx_timeline_event_date", "event_date"),
        sa.Index("idx_timeline_event_type", "event_type"),
        sa.Index("idx_timeline_importance", "importance_score"),
        sa.Index("idx_timeline_financial_impact", "financial_impact"),
        sa.Index("idx_timeline_has_contradiction", "has_contradiction"),
    )


class FinancialEvent(Base):
    """Model representing financial events in a timeline."""
    
    __tablename__ = "financial_events"
    
    id = sa.Column(sa.Integer, primary_key=True)
    matter_id = sa.Column(sa.Integer, sa.ForeignKey("matters.id"), nullable=False)
    timeline_event_id = sa.Column(sa.Integer, sa.ForeignKey("timeline_events.id"), nullable=False)
    document_id = sa.Column(sa.Integer, sa.ForeignKey("documents.id"), nullable=False)
    
    # Financial data
    amount = sa.Column(sa.Float, nullable=False)
    amount_description = sa.Column(sa.String, nullable=True)
    currency = sa.Column(sa.String, default="USD", nullable=False)
    
    # Event categorization
    event_type = sa.Column(sa.String, nullable=False)  # "change_order", "payment", "claim", "credit", etc.
    category = sa.Column(sa.String, nullable=True)  # For custom categorization
    
    # Date information
    event_date = sa.Column(sa.Date, nullable=True, index=True)
    effective_date = sa.Column(sa.Date, nullable=True)  # When the financial impact takes effect
    
    # For tracking running balances
    is_additive = sa.Column(sa.Boolean, default=True)  # Whether the amount adds to or subtracts from the running total
    running_total = sa.Column(sa.Float, nullable=True)  # Running total at the time of this event
    
    # Source information
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    confidence = sa.Column(sa.Float, default=1.0)
    
    # Relationships
    matter = relationship("Matter")
    document = relationship("Document")
    timeline_event = relationship("TimelineEvent", back_populates="financial_events")
    
    # Add indices
    __table_args__ = (
        sa.Index("idx_financial_event_date", "event_date"),
        sa.Index("idx_financial_event_amount", "amount"),
        sa.Index("idx_financial_event_type", "event_type"),
        sa.Index("idx_financial_running_total", "running_total"),
    )


def get_top_chunks_by_similarity(
    vector_ids: List[int], top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get the top chunks by similarity score."""
    from .utils import console
    console.log(f"get_top_chunks_by_similarity called with {len(vector_ids)} vector IDs and top_k={top_k}")

    if not top_k:
        config = get_config()
        top_k = config.retrieval.TOP_K
    
    # Always use twice the requested top_k to ensure enough documents for reranking
    top_k = top_k * 2
    console.log(f"Will retrieve up to {top_k} chunks from database")

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
    console.log(f"Looking up chunks for vector_ids: {vector_ids[:top_k*2]}")  # Use double the top_k to ensure we have enough for reranking

    with get_session() as session:
        chunks = []
        for faiss_id in vector_ids[:top_k*2]:  # Double top_k to get enough chunks for reranking
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


def save_financial_event(event_data: Dict[str, Any]) -> Optional[int]:
    """Save a financial event to the database.
    
    Args:
        event_data: Dictionary containing financial event data
        
    Returns:
        ID of the created financial event, or None if an error occurred
    """
    with get_session() as session:
        try:
            # Start transaction
            transaction = session.begin_nested()
            
            # Create new financial event
            event = FinancialEvent(**event_data)
            session.add(event)
            session.flush()  # To get the ID
            
            # If this is linked to a timeline event, update its financial impact flag
            if event.timeline_event_id:
                timeline_event = session.query(TimelineEvent).filter(
                    TimelineEvent.id == event.timeline_event_id
                ).first()
                
                if timeline_event:
                    # Update the financial impact information
                    timeline_event.financial_impact = event.amount if event.is_additive else -event.amount
                    timeline_event.financial_impact_description = event.amount_description
                    timeline_event.financial_impact_type = event.event_type
                    
                    session.flush()
            
            # Commit transaction
            transaction.commit()
            
            # Return the ID of the newly created event
            return event.id
            
        except Exception as e:
            # Rollback transaction on error
            if transaction.is_active:
                transaction.rollback()
            print(f"Error saving financial event: {str(e)}")
            return None
        
        # Commit the session
        session.commit()


def get_financial_events(
    matter_id: int,
    event_types: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    timeline_event_id: Optional[int] = None,
    document_id: Optional[int] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    limit: int = 100,
    sort_by: str = "date",
) -> List[Dict[str, Any]]:
    """Get financial events for a matter, with optional filtering.
    
    Args:
        matter_id: ID of the matter
        event_types: Optional list of event types to filter by
        date_from: Optional start date for filtering
        date_to: Optional end date for filtering
        timeline_event_id: Optional timeline event ID to filter by
        document_id: Optional document ID to filter by
        min_amount: Minimum amount to filter by
        max_amount: Maximum amount to filter by
        limit: Maximum number of events to return
        sort_by: Field to sort by ('date', 'amount', 'running_total')
        
    Returns:
        List of financial events matching the criteria
    """
    with get_session() as session:
        query = session.query(FinancialEvent).filter(FinancialEvent.matter_id == matter_id)
        
        # Apply filters
        if event_types:
            query = query.filter(FinancialEvent.event_type.in_(event_types))
        
        if date_from:
            query = query.filter(FinancialEvent.event_date >= date_from)
        
        if date_to:
            query = query.filter(FinancialEvent.event_date <= date_to)
        
        if timeline_event_id:
            query = query.filter(FinancialEvent.timeline_event_id == timeline_event_id)
            
        if document_id:
            query = query.filter(FinancialEvent.document_id == document_id)
        
        if min_amount is not None:
            query = query.filter(FinancialEvent.amount >= min_amount)
        
        if max_amount is not None:
            query = query.filter(FinancialEvent.amount <= max_amount)
        
        # Apply sorting
        if sort_by == "date":
            query = query.order_by(FinancialEvent.event_date.asc())
        elif sort_by == "amount":
            query = query.order_by(FinancialEvent.amount.desc())
        elif sort_by == "running_total":
            query = query.order_by(FinancialEvent.running_total.desc())
        else:
            # Default sort by date
            query = query.order_by(FinancialEvent.event_date.asc())
        
        # Apply limit
        events = query.limit(limit).all()
        
        # Format results
        result = []
        for event in events:
            # Get related information
            document = session.query(Document).filter(Document.id == event.document_id).first()
            timeline_event = session.query(TimelineEvent).filter(TimelineEvent.id == event.timeline_event_id).first()
            
            # Format event data
            event_data = {
                "id": event.id,
                "amount": event.amount,
                "amount_description": event.amount_description,
                "currency": event.currency,
                "event_type": event.event_type,
                "category": event.category,
                "event_date": event.event_date.isoformat() if event.event_date else None,
                "effective_date": event.effective_date.isoformat() if event.effective_date else None,
                "is_additive": event.is_additive,
                "running_total": event.running_total,
                "confidence": event.confidence,
                "document": {
                    "id": document.id,
                    "file_name": document.file_name,
                    "file_path": document.file_path,
                    "document_type": document.document_type,
                } if document else None,
                "timeline_event": {
                    "id": timeline_event.id,
                    "event_date": timeline_event.event_date.isoformat() if timeline_event.event_date else None,
                    "event_type": timeline_event.event_type,
                    "description": timeline_event.description,
                } if timeline_event else None,
            }
            
            result.append(event_data)
        
        return result


def get_current_matter_id() -> Optional[int]:
    """Get ID of the current matter.
    
    Returns:
        The ID of the current matter, or None if no matter is active
    """
    from .config import get_current_matter
    
    matter_name = get_current_matter()
    if not matter_name:
        return None
        
    with get_session() as session:
        matter = session.query(Matter).filter(Matter.name == matter_name).first()
        return matter.id if matter else None


def update_running_totals(matter_id: int) -> bool:
    """Recalculate running totals for all financial events in a matter.
    
    Args:
        matter_id: ID of the matter
        
    Returns:
        True if successful, False otherwise
    """
    with get_session() as session:
        try:
            # Start transaction
            transaction = session.begin_nested()
            
            # Get all financial events for the matter, ordered by date
            events = session.query(FinancialEvent).filter(
                FinancialEvent.matter_id == matter_id
            ).order_by(FinancialEvent.event_date.asc()).all()
            
            running_total = 0.0
            
            # Update running totals
            for event in events:
                amount_effect = event.amount if event.is_additive else -event.amount
                running_total += amount_effect
                event.running_total = running_total
            
            # Commit transaction
            transaction.commit()
            return True
            
        except Exception as e:
            # Rollback transaction on error
            if transaction.is_active:
                transaction.rollback()
            print(f"Error updating running totals: {str(e)}")
            return False
        
        # Commit the session
        session.commit()


def identify_contradictions(
    matter_id: int,
    min_confidence: float = 0.6,
    max_date_diff_days: int = 30,
) -> List[Dict[str, Any]]:
    """Identify potential contradictions between timeline events.
    
    Args:
        matter_id: ID of the matter
        min_confidence: Minimum confidence threshold for events
        max_date_diff_days: Maximum difference in days between events to consider them related
        
    Returns:
        List of identified contradictions
    """
    with get_session() as session:
        # Get all timeline events for the matter with minimum confidence
        events = session.query(TimelineEvent).filter(
            TimelineEvent.matter_id == matter_id,
            TimelineEvent.confidence >= min_confidence
        ).all()
        
        contradictions = []
        
        # Compare each pair of events
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                # Skip self-comparison and already processed pairs
                if i >= j:
                    continue
                
                # Skip events without dates for direct comparison
                if not event1.event_date or not event2.event_date:
                    continue
                
                # Calculate date difference
                date_diff = abs((event1.event_date - event2.event_date).days)
                
                # Skip events too far apart in time
                if date_diff > max_date_diff_days:
                    continue
                
                # Check for contradictions in the same category
                if event1.event_type == event2.event_type:
                    # For now, we'll implement a simple contradiction check
                    # This would be enhanced with NLP in a real implementation
                    
                    # Check for financial contradictions
                    if event1.financial_impact and event2.financial_impact:
                        # If they have significantly different financial impacts for the same event type
                        if abs(event1.financial_impact - event2.financial_impact) > 1000:  # $1000 threshold
                            contradictions.append({
                                "event1_id": event1.id,
                                "event2_id": event2.id,
                                "event1_date": event1.event_date.isoformat(),
                                "event2_date": event2.event_date.isoformat(),
                                "event_type": event1.event_type,
                                "contradiction_type": "financial",
                                "details": f"Financial impact discrepancy: ${event1.financial_impact} vs ${event2.financial_impact}",
                                "date_difference_days": date_diff
                            })
                    
                    # Check for timeline contradictions (e.g., different dates for same milestone)
                    if event1.event_type in ("project_start", "project_completion") and date_diff > 7:  # 1 week threshold
                        contradictions.append({
                            "event1_id": event1.id,
                            "event2_id": event2.id,
                            "event1_date": event1.event_date.isoformat(),
                            "event2_date": event2.event_date.isoformat(),
                            "event_type": event1.event_type,
                            "contradiction_type": "timeline",
                            "details": f"Date discrepancy for {event1.event_type}: {event1.event_date.isoformat()} vs {event2.event_date.isoformat()}",
                            "date_difference_days": date_diff
                        })
        
        return contradictions


def save_contradiction(
    event1_id: int,
    event2_id: int,
    contradiction_details: str,
) -> bool:
    """Save contradiction information to the database.
    
    Args:
        event1_id: ID of the first event
        event2_id: ID of the second event
        contradiction_details: Details about the contradiction
        
    Returns:
        True if successful, False otherwise
    """
    with get_session() as session:
        try:
            # Start transaction
            transaction = session.begin_nested()
            
            # Get the events
            event1 = session.query(TimelineEvent).filter(TimelineEvent.id == event1_id).first()
            event2 = session.query(TimelineEvent).filter(TimelineEvent.id == event2_id).first()
            
            if not event1 or not event2:
                return False
            
            # Update both events with contradiction information
            event1.has_contradiction = True
            event1.contradiction_details = contradiction_details
            event1.contradicting_event_id = event2.id
            
            event2.has_contradiction = True
            event2.contradiction_details = contradiction_details
            event2.contradicting_event_id = event1.id
            
            # Commit transaction
            transaction.commit()
            return True
            
        except Exception as e:
            # Rollback transaction on error
            if transaction.is_active:
                transaction.rollback()
            print(f"Error saving contradiction: {str(e)}")
            return False
        
        # Commit the session
        session.commit()


def get_financial_summary(
    matter_id: int,
    event_types: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Dict[str, Any]:
    """Get financial summary for a matter.
    
    Args:
        matter_id: ID of the matter
        event_types: Optional list of event types to filter by
        date_from: Optional start date for filtering
        date_to: Optional end date for filtering
        
    Returns:
        Dictionary containing financial summary data
    """
    with get_session() as session:
        query = session.query(FinancialEvent).filter(FinancialEvent.matter_id == matter_id)
        
        # Apply filters
        if event_types:
            query = query.filter(FinancialEvent.event_type.in_(event_types))
        
        if date_from:
            query = query.filter(FinancialEvent.event_date >= date_from)
        
        if date_to:
            query = query.filter(FinancialEvent.event_date <= date_to)
        
        events = query.order_by(FinancialEvent.event_date.asc()).all()
        
        # Calculate summary statistics
        total_amount = 0.0
        total_positive = 0.0
        total_negative = 0.0
        event_type_totals = {}
        category_totals = {}
        monthly_totals = {}
        
        for event in events:
            amount = event.amount if event.is_additive else -event.amount
            total_amount += amount
            
            if amount > 0:
                total_positive += amount
            else:
                total_negative += amount
            
            # Track totals by event type
            event_type = event.event_type
            if event_type not in event_type_totals:
                event_type_totals[event_type] = 0.0
            event_type_totals[event_type] += amount
            
            # Track totals by category
            category = event.category or "uncategorized"
            if category not in category_totals:
                category_totals[category] = 0.0
            category_totals[category] += amount
            
            # Track monthly totals
            if event.event_date:
                month_key = f"{event.event_date.year}-{event.event_date.month:02d}"
                if month_key not in monthly_totals:
                    monthly_totals[month_key] = 0.0
                monthly_totals[month_key] += amount
        
        # Get the latest running total
        latest_total = events[-1].running_total if events else 0.0
        
        # Build summary
        summary = {
            "total_amount": total_amount,
            "total_positive": total_positive,
            "total_negative": total_negative,
            "latest_running_total": latest_total,
            "event_type_totals": event_type_totals,
            "category_totals": category_totals,
            "monthly_totals": monthly_totals,
            "event_count": len(events),
        }
        
        return summary


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
