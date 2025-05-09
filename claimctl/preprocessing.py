"""
Preprocessing module for the claim-assistant system.

This module handles preprocessing of PDF files, including document segmentation
for large PDFs containing multiple logical documents. It integrates with the
document_segmentation module and the existing ingestion pipeline.
"""

import os
import logging
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Tuple
import fitz  # PyMuPDF

from claimctl.document_segmentation import process_pdf_for_segmentation
from claimctl.database import get_db_connection

# Setup logging
logger = logging.getLogger(__name__)

def preprocess_pdf(pdf_path: str, output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main preprocessing function that handles document segmentation.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save processed files
        config: Configuration parameters
    
    Returns:
        Dict with results of preprocessing including paths to split documents
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get document segmentation settings from config
    seg_config = {
        'SEGMENT_SIZE': config.get('document_segmentation', {}).get('SEGMENT_SIZE', 500),
        'SEGMENT_STRIDE': config.get('document_segmentation', {}).get('SEGMENT_STRIDE', 100),
        'THRESHOLD_MULTIPLIER': config.get('document_segmentation', {}).get('THRESHOLD_MULTIPLIER', 1.5),
        'MIN_CONFIDENCE': config.get('document_segmentation', {}).get('MIN_CONFIDENCE', 0.3),
        'MIN_DOCUMENT_LENGTH': config.get('document_segmentation', {}).get('MIN_DOCUMENT_LENGTH', 1000),
        'MIN_BOUNDARY_DISTANCE': config.get('document_segmentation', {}).get('MIN_BOUNDARY_DISTANCE', 2000),
        'VISUALIZE': config.get('document_segmentation', {}).get('VISUALIZE', False)
    }
    
    # Process the PDF for segmentation
    logger.info(f"Preprocessing PDF: {pdf_path}")
    
    try:
        # Run document segmentation
        result = process_pdf_for_segmentation(
            pdf_path=pdf_path,
            output_dir=output_dir,
            config=seg_config
        )
        
        # Add file hashes for tracking
        result['original_pdf_hash'] = calculate_file_hash(pdf_path)
        
        for doc in result['documents']:
            doc['hash'] = calculate_file_hash(doc['path'])
        
        logger.info(f"Successfully preprocessed {pdf_path}, found {result['documents_found']} documents")
        return result
        
    except Exception as e:
        logger.error(f"Error preprocessing {pdf_path}: {str(e)}")
        raise

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        SHA-256 hash of the file
    """
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def store_document_relationship(
    matter_id: str,
    original_pdf: str,
    derived_pdf: str,
    rel_type: str = 'segment',
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    doc_type: Optional[str] = None,
    confidence: Optional[float] = None,
    original_hash: Optional[str] = None,
    derived_hash: Optional[str] = None
) -> int:
    """
    Store relationship between original PDF and a derived document in the database.
    
    Args:
        matter_id: ID of the matter
        original_pdf: Path to the original PDF
        derived_pdf: Path to the derived PDF
        rel_type: Relationship type ('segment', 'extract', etc.)
        start_page: Start page of the segment in the original PDF
        end_page: End page of the segment in the original PDF
        doc_type: Document type classification
        confidence: Confidence score for document type
        original_hash: Hash of the original PDF file
        derived_hash: Hash of the derived PDF file
    
    Returns:
        ID of the inserted relationship record
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Calculate hashes if not provided
    if not original_hash:
        original_hash = calculate_file_hash(original_pdf)
    if not derived_hash:
        derived_hash = calculate_file_hash(derived_pdf)
    
    c.execute(
        """
        INSERT INTO document_relationships (
            matter_id, 
            original_pdf_path, 
            derived_pdf_path, 
            original_pdf_hash, 
            derived_pdf_hash, 
            relationship_type, 
            start_page, 
            end_page,
            document_type,
            confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            matter_id,
            original_pdf,
            derived_pdf,
            original_hash,
            derived_hash,
            rel_type,
            start_page,
            end_page,
            doc_type,
            confidence
        )
    )
    
    relationship_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return relationship_id

def get_derived_documents(matter_id: str, original_pdf: str) -> List[Dict[str, Any]]:
    """
    Get all derived documents from an original PDF.
    
    Args:
        matter_id: ID of the matter
        original_pdf: Path to the original PDF
    
    Returns:
        List of derived document records
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute(
        """
        SELECT 
            id, 
            derived_pdf_path, 
            relationship_type, 
            start_page, 
            end_page, 
            document_type, 
            confidence, 
            created_at
        FROM document_relationships
        WHERE matter_id = ? AND original_pdf_path = ?
        ORDER BY start_page
        """,
        (matter_id, original_pdf)
    )
    
    results = [{
        'id': row[0],
        'path': row[1],
        'relationship_type': row[2],
        'start_page': row[3],
        'end_page': row[4],
        'document_type': row[5],
        'confidence': row[6],
        'created_at': row[7]
    } for row in c.fetchall()]
    
    conn.close()
    return results

def get_original_document(matter_id: str, derived_pdf: str) -> Optional[Dict[str, Any]]:
    """
    Get original document information for a derived document.
    
    Args:
        matter_id: ID of the matter
        derived_pdf: Path to the derived PDF
    
    Returns:
        Dict with original document information or None if not found
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute(
        """
        SELECT 
            id, 
            original_pdf_path, 
            relationship_type, 
            start_page, 
            end_page, 
            document_type, 
            confidence, 
            created_at
        FROM document_relationships
        WHERE matter_id = ? AND derived_pdf_path = ?
        LIMIT 1
        """,
        (matter_id, derived_pdf)
    )
    
    row = c.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        'id': row[0],
        'path': row[1],
        'relationship_type': row[2],
        'start_page': row[3],
        'end_page': row[4],
        'document_type': row[5],
        'confidence': row[6],
        'created_at': row[7]
    }

def prepare_db_schema() -> None:
    """
    Prepare database schema for document relationships.
    Creates table if it doesn't exist.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create document_relationships table if it doesn't exist
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS document_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matter_id TEXT NOT NULL,
            original_pdf_path TEXT NOT NULL,
            derived_pdf_path TEXT NOT NULL,
            original_pdf_hash TEXT,
            derived_pdf_hash TEXT,
            relationship_type TEXT NOT NULL,
            start_page INTEGER,
            end_page INTEGER,
            document_type TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    
    # Create indices for faster lookups
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_doc_rel_original 
        ON document_relationships(original_pdf_path, matter_id)
        """
    )
    
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_doc_rel_derived 
        ON document_relationships(derived_pdf_path, matter_id)
        """
    )
    
    conn.commit()
    conn.close()
    
    logger.info("Document relationships database schema prepared")

def process_large_pdf(
    pdf_path: str, 
    output_dir: str, 
    matter_id: str, 
    config: Dict[str, Any]
) -> List[str]:
    """
    Process a large PDF file by segmenting it into logical documents
    and preparing them for ingestion.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save processed files 
        matter_id: ID of the current matter
        config: Configuration parameters
    
    Returns:
        List of paths to the split document PDFs
    """
    # Create output directory with unique name based on PDF filename
    pdf_basename = os.path.basename(pdf_path).replace('.pdf', '')
    doc_output_dir = os.path.join(output_dir, pdf_basename)
    os.makedirs(doc_output_dir, exist_ok=True)
    
    # Ensure database schema is prepared
    prepare_db_schema()
    
    # Run preprocessing to segment the PDF
    result = preprocess_pdf(pdf_path, doc_output_dir, config)
    
    # Store document relationships in the database
    for doc in result['documents']:
        store_document_relationship(
            matter_id=matter_id,
            original_pdf=pdf_path,
            derived_pdf=doc['path'],
            rel_type='segment',
            start_page=doc['start_page'],
            end_page=doc['end_page'],
            doc_type=doc['doc_type'],
            confidence=doc['confidence'],
            original_hash=result['original_pdf_hash'],
            derived_hash=doc['hash']
        )
    
    # Return paths to the split documents
    return [doc['path'] for doc in result['documents']]

def should_preprocess_pdf(pdf_path: str, config: Dict[str, Any]) -> bool:
    """
    Determine if a PDF file should be preprocessed based on size and pages.
    
    Args:
        pdf_path: Path to the PDF file
        config: Configuration parameters
    
    Returns:
        True if the PDF should be preprocessed, False otherwise
    """
    # Get thresholds from config
    pages_threshold = config.get('document_segmentation', {}).get('PAGES_THRESHOLD', 50)
    size_threshold = config.get('document_segmentation', {}).get('SIZE_THRESHOLD', 10_000_000)  # 10MB
    
    # Check file size
    file_size = os.path.getsize(pdf_path)
    
    # Check number of pages
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        logger.error(f"Error checking PDF pages for {pdf_path}: {str(e)}")
        return False
    
    # Determine if the PDF meets preprocessing criteria
    return (num_pages > pages_threshold) or (file_size > size_threshold)