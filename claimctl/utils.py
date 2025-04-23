"""Utility functions for claim-assistant."""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import pytesseract
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn

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
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # 300 dpi
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


def save_page_image(doc: fitz.Document, page_num: int) -> Dict[str, str]:
    """Save page images (full and thumbnail) and return their paths."""
    config = get_config()
    page = doc[page_num]

    # Generate a filename based on the PDF name and page number
    pdf_name = Path(doc.name).stem
    pages_dir = Path(config.paths.DATA_DIR) / "pages"
    thumbs_dir = Path(config.paths.DATA_DIR) / "thumbnails"
    pages_dir.mkdir(exist_ok=True, parents=True)
    thumbs_dir.mkdir(exist_ok=True, parents=True)

    # Create a high-quality image for full view
    full_pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # 300 dpi
    img_path = pages_dir / f"{pdf_name}_{page_num+1:04d}.png"  # 1-indexed for humans
    full_pix.save(str(img_path))

    # Create a thumbnail version
    thumb_pix = page.get_pixmap(matrix=fitz.Matrix(72 / 72, 72 / 72))  # 72 dpi (screen resolution)
    thumb_path = thumbs_dir / f"{pdf_name}_{page_num+1:04d}_thumb.png"
    thumb_pix.save(str(thumb_path))

    return {
        "image_path": str(img_path),
        "thumbnail_path": str(thumb_path)
    }


def preprocess_image_for_ocr(image_path: str) -> str:
    """Apply image preprocessing for better OCR results.

    Applies:
    - Contrast enhancement
    - Deskew (straighten text)
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter

        # Load image
        img = Image.open(image_path)

        # Convert to grayscale if not already
        if img.mode != "L":
            img = img.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)  # Increase contrast

        # Apply slight sharpening
        img = img.filter(ImageFilter.SHARPEN)

        # Save preprocessed image
        preprocessed_path = image_path.replace(".png", "_preprocessed.png")
        img.save(preprocessed_path)

        # Attempt deskew using OpenCV
        try:
            # Load with OpenCV
            cv_img = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)

            # Threshold to create binary image
            _, binary = cv2.threshold(cv_img, 128, 255, cv2.THRESH_BINARY_INV)

            # Find all contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # Find rotated rectangle for largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]

                # Determine if we need to correct the angle
                if angle < -45:
                    angle = 90 + angle

                # Only deskew if angle is significant
                if abs(angle) > 0.5:
                    (h, w) = cv_img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(
                        cv_img,
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    cv2.imwrite(preprocessed_path, rotated)
        except Exception as e:
            console.log(f"[yellow]Deskew failed: {str(e)}")

        return preprocessed_path

    except ImportError as e:
        console.log(
            f"[yellow]Image preprocessing skipped (missing libraries): {str(e)}"
        )
        return image_path
    except Exception as e:
        console.log(f"[yellow]Image preprocessing error: {str(e)}")
        return image_path


def run_ocr_on_page(doc: fitz.Document, page_num: int) -> str:
    """Run OCR on a PDF page using Tesseract."""
    config = get_config()
    page = doc[page_num]

    # Save page as a temporary PNG
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # 300 dpi
    cache_dir = Path(config.paths.DATA_DIR) / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    temp_img_path = cache_dir / f"temp_ocr_{get_page_hash(doc.name, page_num)}.png"
    pix.save(str(temp_img_path))

    # Preprocess image
    preprocessed_img_path = preprocess_image_for_ocr(str(temp_img_path))

    # Run OCR with custom configuration
    # Different settings based on document content (detect if it's primarily text or images)
    is_text_heavy = len(extract_text_from_page(doc, page_num).strip()) > 50

    if is_text_heavy:
        # For text-heavy documents, optimize for text recognition
        config_params = "--oem 3 --psm 3"  # Default Tesseract OCR Engine mode with automatic page segmentation
    else:
        # For image-heavy documents with sparse text
        config_params = "--oem 3 --psm 6"  # Assume a single uniform block of text

    text = pytesseract.image_to_string(preprocessed_img_path, config=config_params)

    # Clean up temp files
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    if preprocessed_img_path != str(temp_img_path) and os.path.exists(
        preprocessed_img_path
    ):
        os.remove(preprocessed_img_path)

    return text


def process_page(doc: fitz.Document, page_num: int) -> Dict[str, Any]:
    """Process a single page: extract text, save image and thumbnail."""
    # Extract text using PyMuPDF
    text = extract_text_from_page(doc, page_num)

    # Increased character threshold from 20 to 150
    if len(text.strip()) < 150:
        console.log(f"Using OCR fallback for page {page_num+1}")
        text = run_ocr_on_page(doc, page_num)

    # Save page as image and thumbnail
    image_paths = save_page_image(doc, page_num)

    return {
        "text": text,
        "image_path": image_paths["image_path"],
        "thumbnail_path": image_paths["thumbnail_path"],
        "page_hash": get_page_hash(doc.name, page_num),
    }


def highlight_text(text: str, query: str) -> str:
    """Highlight query terms in text with formatting for display.
    
    Args:
        text: The original text
        query: The search query
        
    Returns:
        Text with query terms wrapped in rich formatting
    """
    if not query or not text:
        return text
        
    # Break the query into individual terms, ignoring common words
    stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "of", "is", "are", "was", "were", "to", "for"}
    query_terms = set()
    
    # Add phrases (multi-word terms in quotes)
    phrase_matches = re.findall(r'"([^"]+)"', query)
    for phrase in phrase_matches:
        query_terms.add(phrase.lower())
        # Remove phrases from the query for individual word processing
        query = query.replace(f'"{phrase}"', '')
    
    # Add individual significant words
    for word in re.findall(r'\b\w{3,}\b', query.lower()):
        if word not in stop_words:
            query_terms.add(word)
    
    # Sort terms by length (longest first) to avoid partial matches
    sorted_terms = sorted(query_terms, key=len, reverse=True)
    
    # Highlight each term with rich formatting
    highlighted_text = text
    for term in sorted_terms:
        # Create a regex that matches whole words and is case-insensitive
        if len(term.split()) > 1:  # Multi-word phrase
            pattern = re.compile(re.escape(term), re.IGNORECASE)
        else:  # Single word - match whole words only
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
        # Replace with highlighted version
        highlighted_text = pattern.sub(lambda m: f"[bold yellow]{m.group(0)}[/bold yellow]", highlighted_text)
    
    return highlighted_text


def create_progress(description: str) -> Progress:
    """Create a rich progress bar."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TextColumn("[italic]{task.fields[status]}"),
    )
