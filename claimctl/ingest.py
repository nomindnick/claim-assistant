"""PDF ingestion module for claim-assistant."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import faiss
import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from rich.progress import Progress, TaskID

from .config import get_config, ensure_dirs
from .database import (
    init_database, is_page_processed, save_page_chunk, get_database_engine
)
from .utils import (
    calculate_sha256, get_page_hash, process_page, create_progress, console
)


# Chunk type classification prompt
CHUNK_TYPE_PROMPT = """
You are a construction document analyzer. Your task is to classify this document chunk into one of the following categories:
- Email: Any email correspondence
- ChangeOrder: Documents describing changes to project scope, timeline, or costs
- Invoice: Bills, receipts, financial documents
- Photo: Document is primarily a photograph with minimal text
- ContractClause: Excerpts from the contract
- Other: Anything that doesn't fit the above categories

Document text:
{text}

Provide ONLY the category name as your response, nothing else.
"""

# Date extraction regex patterns
DATE_PATTERNS = [
    r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",  # MM/DD/YYYY or similar
    r"(?:\w{3,9}\s+\d{1,2},?\s+\d{4})",  # Month DD, YYYY
    r"(?:\d{1,2}\s+\w{3,9},?\s+\d{4})",  # DD Month YYYY
]

# Document ID extraction regex patterns
ID_PATTERNS = [
    r"(?:Change Order|CO)\s+#?\d+",  # Change Order #123
    r"(?:Invoice|INV)\s+#?\d+",  # Invoice #123
    r"(?:Document|Doc)\s+#?\d+",  # Document #123
    r"(?:Contract|Agreement)\s+#?[\w-]+",  # Contract #A-123
]


def extract_dates(text: str) -> List[str]:
    """Extract dates from text using regex patterns."""
    dates = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates


def extract_document_ids(text: str) -> List[str]:
    """Extract document IDs from text using regex patterns."""
    ids = []
    for pattern in ID_PATTERNS:
        matches = re.findall(pattern, text)
        ids.extend(matches)
    return ids


def classify_chunk(text: str) -> str:
    """Classify the type of document chunk using GPT-4o-mini."""
    config = get_config()
    
    # Temporary debug mode to bypass API calls
    debug_mode = False
    
    if debug_mode:
        console.log("DEBUG MODE: Skipping API call for classification")
        # Look for keywords to classify
        if "change order" in text.lower() or "co #" in text.lower():
            return "ChangeOrder"
        elif "@" in text.lower() and ("from:" in text.lower() or "to:" in text.lower()):
            return "Email"
        elif "invoice" in text.lower() or "$" in text:
            return "Invoice"
        elif len(text.strip()) < 100:  # Assume it's a photo with minimal text
            return "Photo"
        elif "clause" in text.lower() or "section" in text.lower() or "agreement" in text.lower():
            return "ContractClause"
        else:
            return "Other"
    
    try:
        client = OpenAI(api_key=config.openai.API_KEY)
        response = client.chat.completions.create(
            model=config.openai.MODEL,
            messages=[
                {"role": "system", "content": "You are a construction document classifier assistant."},
                {"role": "user", "content": CHUNK_TYPE_PROMPT.format(text=text[:1000])},  # Truncate text
            ],
            temperature=0.0,
            max_tokens=20,
        )
        
        chunk_type = response.choices[0].message.content.strip()
        # Normalize response to match one of our categories
        valid_types = ["Email", "ChangeOrder", "Invoice", "Photo", "ContractClause", "Other"]
        
        for valid_type in valid_types:
            if valid_type.lower() in chunk_type.lower():
                return valid_type
        
        return "Other"
    except Exception as e:
        console.log(f"[bold red]Error classifying chunk: {str(e)}")
        return "Other"


def create_or_load_faiss_index(dim: int = 1536) -> faiss.IndexFlatL2:
    """Create a new FAISS index or load an existing one."""
    config = get_config()
    index_path = Path(config.paths.INDEX_DIR) / "faiss.idx"
    
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            console.log(f"Loaded existing FAISS index with {index.ntotal} vectors")
            return index
        except Exception as e:
            console.log(f"Error loading FAISS index: {e}. Creating a new one.")
    
    # Create a new index
    index = faiss.IndexFlatL2(dim)  # L2 distance
    console.log(f"Created new FAISS index with dimension {dim}")
    return index


def save_faiss_index(index: faiss.IndexFlatL2) -> None:
    """Save the FAISS index to disk."""
    config = get_config()
    index_path = Path(config.paths.INDEX_DIR) / "faiss.idx"
    faiss.write_index(index, str(index_path))
    console.log(f"Saved FAISS index with {index.ntotal} vectors")


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    config = get_config()
    
    # Temporary debug mode to bypass API calls
    debug_mode = False
    
    if debug_mode:
        console.log("DEBUG MODE: Generating fake embeddings")
        # Generate deterministic fake embeddings based on text content
        # This allows us to test the pipeline without API calls
        embeddings = []
        for text in texts:
            # Create a deterministic "embedding" based on hash of text
            text_hash = hash(text) % 10000
            np.random.seed(text_hash)
            fake_embedding = np.random.randn(1536).astype(np.float32)
            # Normalize to unit length like real embeddings
            fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)
            embeddings.append(fake_embedding)
        return np.array(embeddings, dtype=np.float32)
    
    try:
        embed_model = OpenAIEmbeddings(
            openai_api_key=config.openai.API_KEY,
            model=config.openai.EMBED_MODEL,
        )
        
        embeddings = embed_model.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        console.log(f"[bold red]Error generating embeddings: {str(e)}")
        # Return zeros as fallback
        return np.zeros((len(texts), 1536), dtype=np.float32)


def process_pdf(pdf_path: Path, progress: Progress, task_id: TaskID) -> None:
    """Process a PDF file: extract text, metadata, generate embeddings."""
    config = get_config()
    console.log(f"Processing {pdf_path}")
    
    # Initialize document
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        console.log(f"[bold red]Error opening {pdf_path}: {str(e)}")
        progress.update(task_id, advance=1, status=f"Error: {str(e)}")
        return
    
    # Get/create FAISS index
    index = create_or_load_faiss_index()
    
    # Process each page
    total_pages = len(doc)
    progress.update(task_id, total=total_pages, status="Starting")
    
    for page_num in range(total_pages):
        progress.update(task_id, advance=0, status=f"Page {page_num+1}/{total_pages}")
        
        # Generate unique hash for this page
        page_hash = get_page_hash(pdf_path, page_num)
        
        # Skip if already processed
        if is_page_processed(page_hash):
            console.log(f"Skipping page {page_num+1} (already processed)")
            progress.update(task_id, advance=1, status=f"Skipped (duplicate)")
            continue
        
        # Process page
        try:
            page_data = process_page(doc, page_num)
            text = page_data["text"]
            
            # Extract metadata
            dates = extract_dates(text)
            doc_ids = extract_document_ids(text)
            
            # Classify chunk
            chunk_type = classify_chunk(text)
            
            # Prepare DB entry
            chunk_data = {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "page_num": page_num + 1,  # 1-indexed for humans
                "page_hash": page_hash,
                "image_path": page_data["image_path"],
                "text": text,
                "chunk_type": chunk_type,
                "doc_date": None,  # Will parse dates later if needed
                "doc_id": doc_ids[0] if doc_ids else None,
            }
            
            # Generate embedding
            embedding = get_embeddings([text])[0]
            
            # Add to FAISS index
            index.add(np.array([embedding], dtype=np.float32))
            
            # Save to database
            save_page_chunk(chunk_data)
            
            progress.update(task_id, advance=1, status=f"Processed {chunk_type}")
            
        except Exception as e:
            console.log(f"[bold red]Error processing page {page_num+1}: {str(e)}")
            progress.update(task_id, advance=1, status=f"Error: {str(e)}")
    
    # Close document
    doc.close()
    
    # Save updated index
    save_faiss_index(index)


def ingest_pdfs(pdf_paths: List[Path]) -> None:
    """Ingest a list of PDF files."""
    # Ensure directories exist
    ensure_dirs()
    
    # Initialize database
    init_database()
    
    # Create progress context
    with create_progress("Ingesting PDFs") as progress:
        # Create a task for each PDF
        pdf_tasks = {}
        for pdf_path in pdf_paths:
            task_id = progress.add_task(f"Processing {pdf_path.name}", total=1, status="Waiting")
            pdf_tasks[pdf_path] = task_id
        
        # Process each PDF
        for pdf_path, task_id in pdf_tasks.items():
            process_pdf(pdf_path, progress, task_id)
    
    console.log("[bold green]Ingestion complete!")
