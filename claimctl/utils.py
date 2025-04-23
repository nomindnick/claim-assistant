"""Utility functions for claim-assistant."""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import pytesseract
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskID

from .config import get_config

# Initialize rich console
console = Console()


def calculate_sha256(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_page_hash(pdf_path: Union[str, Path], page_num: int) -> str:
    """Generate a unique hash for a specific PDF page."""
    file_hash = calculate_sha256(pdf_path)
    # Combine file hash with page number for a unique page identifier
    page_identifier = f"{file_hash}_{page_num}"
    return hashlib.sha256(page_identifier.encode()).hexdigest()


def extract_text_from_page(doc: fitz.Document, page_num: int) -> str:
    """Extract text from a PDF page using PyMuPDF."""
    page = doc[page_num]
    return page.get_text("text")


def run_ocr_on_page(doc: fitz.Document, page_num: int) -> str:
    """Run OCR on a PDF page using Tesseract."""
    config = get_config()
    page = doc[page_num]
    
    # Save page as a temporary PNG
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 dpi
    cache_dir = Path(config.paths.DATA_DIR) / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    temp_img_path = cache_dir / f"temp_ocr_{get_page_hash(doc.name, page_num)}.png"
    pix.save(str(temp_img_path))
    
    # Run OCR
    text = pytesseract.image_to_string(str(temp_img_path))
    
    # Clean up temp file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    return text


def save_page_image(doc: fitz.Document, page_num: int) -> str:
    """Save a page as a PNG image and return the path."""
    config = get_config()
    page = doc[page_num]
    
    # Generate a filename based on the PDF name and page number
    pdf_name = Path(doc.name).stem
    pages_dir = Path(config.paths.DATA_DIR) / "pages"
    pages_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a high-quality image
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 dpi
    img_path = pages_dir / f"{pdf_name}_{page_num+1:04d}.png"  # 1-indexed for humans
    pix.save(str(img_path))
    
    return str(img_path)


def process_page(doc: fitz.Document, page_num: int) -> Dict[str, str]:
    """Process a single page: extract text, save image."""
    # Extract text using PyMuPDF
    text = extract_text_from_page(doc, page_num)
    
    # If less than 20 characters found, run OCR
    if len(text.strip()) < 20:
        console.log(f"Using OCR fallback for page {page_num+1}")
        text = run_ocr_on_page(doc, page_num)
    
    # Save page as image
    img_path = save_page_image(doc, page_num)
    
    return {
        "text": text,
        "image_path": img_path,
        "page_hash": get_page_hash(doc.name, page_num),
    }


def create_progress(description: str) -> Progress:
    """Create a rich progress bar."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TextColumn("[italic]{task.fields[status]}"),
    )
