"""Utility functions for claim-assistant."""

import hashlib
import os
import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    thumb_pix = page.get_pixmap(
        matrix=fitz.Matrix(72 / 72, 72 / 72)
    )  # 72 dpi (screen resolution)
    thumb_path = thumbs_dir / f"{pdf_name}_{page_num+1:04d}_thumb.png"
    thumb_pix.save(str(thumb_path))

    return {"image_path": str(img_path), "thumbnail_path": str(thumb_path)}


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
    stop_words = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "of",
        "is",
        "are",
        "was",
        "were",
        "to",
        "for",
    }
    query_terms = set()

    # Add phrases (multi-word terms in quotes)
    phrase_matches = re.findall(r'"([^"]+)"', query)
    for phrase in phrase_matches:
        query_terms.add(phrase.lower())
        # Remove phrases from the query for individual word processing
        query = query.replace(f'"{phrase}"', "")

    # Add individual significant words
    for word in re.findall(r"\b\w{3,}\b", query.lower()):
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
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)

        # Replace with highlighted version
        highlighted_text = pattern.sub(
            lambda m: f"[bold yellow]{m.group(0)}[/bold yellow]", highlighted_text
        )

    return highlighted_text


def create_progress(description: str) -> Progress:
    """Create a rich progress bar."""
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
    )
    # Add description to show in the progress bar
    return progress


def export_response_as_pdf(
    question: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    scores: List[float],
) -> str:
    """Export the LLM response and document images as a PDF.
    
    Args:
        question: The user's question
        answer: The LLM's answer
        chunks: Document chunks used to generate the answer
        scores: Relevance scores for each chunk
        
    Returns:
        The path to the exported PDF file
    """
    try:
        import datetime
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, 
            Paragraph, 
            Spacer, 
            Table, 
            TableStyle, 
            Image
        )
        from reportlab.lib import colors
        
        # Create the Ask_Exports directory if it doesn't exist
        exports_dir = Path("./Ask_Exports")
        exports_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(exports_dir / f"response_{timestamp}.pdf")
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading1"]
        normal_style = styles["Normal"]
        
        # Create content elements list
        elements = []
        
        # Add title
        elements.append(Paragraph("Claim Assistant Query Response", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated on: {date_str}", normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add question
        elements.append(Paragraph("Question:", heading_style))
        elements.append(Paragraph(question, normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add answer with proper formatting for markdown content
        elements.append(Paragraph("Answer:", heading_style))
        
        # Convert markdown to HTML-like formatting that ReportLab can handle
        formatted_answer = answer
        # Handle bold text
        formatted_answer = formatted_answer.replace("**", "<b>", 1)
        while "**" in formatted_answer:
            formatted_answer = formatted_answer.replace("**", "</b>", 1)
            if "**" in formatted_answer:
                formatted_answer = formatted_answer.replace("**", "<b>", 1)
        
        # Handle citations [Doc X, p.Y]
        import re
        citation_pattern = r'\[Doc \d+, p\.\d+\]'
        formatted_answer = re.sub(
            citation_pattern,
            lambda m: f'<b>{m.group(0)}</b>',
            formatted_answer
        )
        
        # Handle paragraphs
        paragraphs = formatted_answer.split('\n\n')
        for para in paragraphs:
            if para.strip():
                elements.append(Paragraph(para, normal_style))
                elements.append(Spacer(1, 0.1*inch))
        
        # Add sources section
        elements.append(Spacer(1, 0.15*inch))
        elements.append(Paragraph("Referenced Documents:", heading_style))
        
        # Create a table for documents
        data = [["#", "File", "Page", "Type", "Date", "Relevance"]]
        
        # Add each source to the table
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            data.append([
                str(i + 1),
                chunk["file_name"],
                str(chunk["page_num"]),
                chunk.get("chunk_type", ""),
                str(chunk.get("doc_date", "")),
                f"{score:.2f}"
            ])
        
        # Create and style the table
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add document images
        elements.append(Paragraph("Document Images:", heading_style))
        
        # For each chunk that has an image, add it to the PDF
        for i, chunk in enumerate(chunks):
            if chunk.get("image_path") and os.path.exists(chunk["image_path"]):
                elements.append(Paragraph(
                    f"Source {i+1}: {chunk['file_name']} (Page {chunk['page_num']})",
                    styles["Heading2"]
                ))
                
                # Add image with appropriate scaling
                img = Image(chunk["image_path"], width=6*inch, height=8*inch, kind='proportional')
                elements.append(img)
                elements.append(Spacer(1, 0.25*inch))
        
        # Build the PDF
        doc.build(elements)
        
        return filename
    
    except ImportError as e:
        console.print(f"[bold red]Error: Required PDF export libraries are missing. {str(e)}")
        console.print("[bold yellow]Try installing with: pip install reportlab")
        return ""
    except Exception as e:
        console.print(f"[bold red]Error exporting response as PDF: {str(e)}")
        return ""


def export_timeline_as_pdf(
    timeline_data: Dict[str, Any],
    matter_name: str,
    filters: Dict[str, Any] = None,
) -> str:
    """Export a timeline as a PDF.
    
    Args:
        timeline_data: Timeline data from generate_claim_timeline
        matter_name: Name of the matter
        filters: Dictionary of filters applied to the timeline
        
    Returns:
        The path to the exported PDF file
    """
    try:
        import datetime
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, 
            Paragraph, 
            Spacer, 
            Table, 
            TableStyle,
            PageBreak
        )
        from reportlab.lib import colors
        
        # Create the Timeline_Exports directory if it doesn't exist
        exports_dir = Path("./Timeline_Exports")
        exports_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(exports_dir / f"timeline_{matter_name}_{timestamp}.pdf")
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading1"]
        heading2_style = styles["Heading2"]
        normal_style = styles["Normal"]
        
        # Create custom styles
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue
        )
        
        # Create content elements list
        elements = []
        
        # Add title
        elements.append(Paragraph(f"Claim Timeline: {matter_name}", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated on: {date_str}", normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add filter information if provided
        if filters:
            filter_text = "Filters applied: "
            filter_parts = []
            
            if filters.get("event_types"):
                filter_parts.append(f"Event types: {', '.join(filters['event_types'])}")
            if filters.get("date_from"):
                filter_parts.append(f"From: {filters['date_from']}")
            if filters.get("date_to"):
                filter_parts.append(f"To: {filters['date_to']}")
            if filters.get("min_importance"):
                filter_parts.append(f"Min importance: {filters['min_importance']}")
            if filters.get("min_confidence"):
                filter_parts.append(f"Min confidence: {filters['min_confidence']}")
                
            if filter_parts:
                filter_text += "; ".join(filter_parts)
                elements.append(Paragraph(filter_text, normal_style))
                elements.append(Spacer(1, 0.25*inch))
        
        # Add summary
        summary = timeline_data.get("summary", "No timeline summary available.")
        elements.append(Paragraph("Executive Summary", heading_style))
        
        # Split summary into paragraphs
        summary_paragraphs = summary.split('\n\n')
        for para in summary_paragraphs:
            if para.strip():
                elements.append(Paragraph(para.strip(), normal_style))
                elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.25*inch))
        
        # Add timeline events table
        elements.append(Paragraph("Timeline Events", heading_style))
        elements.append(Spacer(1, 0.15*inch))
        
        events = timeline_data.get("events", [])
        if not events:
            elements.append(Paragraph("No timeline events found.", normal_style))
        else:
            # Create event table
            data = [["Date", "Type", "Description", "Document", "Importance"]]
            
            # Add each event to the table
            for event in events:
                event_date = event.get("event_date", "Unknown")
                event_type = event.get("event_type", "other")
                description = event.get("description", "")
                document_name = event.get("document", {}).get("file_name", "Unknown")
                importance = f"{event.get('importance_score', 0):.2f}"
                
                data.append([
                    event_date,
                    event_type,
                    description,
                    document_name,
                    importance
                ])
            
            # Create and style the table
            # Create and style the table with wordwrap for the description column
            table = Table(data, repeatRows=1, colWidths=[1*inch, 1.2*inch, 3*inch, 1.5*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Allow wordwrap for the description column
                ('VALIGN', (2, 1), (2, -1), 'TOP'),
            ]))
            
            elements.append(table)
            
        # Add chronological view
        elements.append(PageBreak())
        elements.append(Paragraph("Chronological View", heading_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Group events by year and month
        events_by_month = timeline_data.get("events_by_month", {})
        if not events_by_month:
            elements.append(Paragraph("No timeline events found for chronological view.", normal_style))
        else:
            years = {}
            for event in events:
                if event.get("event_date"):
                    date_obj = datetime.datetime.fromisoformat(event["event_date"]).date()
                    year = date_obj.year
                    month = date_obj.month
                    
                    if year not in years:
                        years[year] = {}
                    
                    if month not in years[year]:
                        years[year][month] = []
                    
                    years[year][month].append(event)
            
            # Sort years and months
            for year in sorted(years.keys()):
                elements.append(Paragraph(f"{year}", heading_style))
                elements.append(Spacer(1, 0.15*inch))
                
                for month in sorted(years[year].keys()):
                    month_name = datetime.date(year, month, 1).strftime("%B")
                    elements.append(Paragraph(f"{month_name}", heading2_style))
                    elements.append(Spacer(1, 0.1*inch))
                    
                    for event in years[year][month]:
                        date_obj = datetime.datetime.fromisoformat(event["event_date"]).date()
                        day = date_obj.day
                        
                        # Format event details
                        event_type = event.get("event_type", "other")
                        description = event.get("description", "")
                        document_name = event.get("document", {}).get("file_name", "Unknown")
                        
                        elements.append(Paragraph(f"{day:02d} {month_name[:3]} - {event_type}", date_style))
                        elements.append(Paragraph(description, normal_style))
                        elements.append(Paragraph(f"Document: {document_name}", normal_style))
                        elements.append(Spacer(1, 0.15*inch))
                
                elements.append(Spacer(1, 0.25*inch))
        
        # Add total count
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Total events: {len(events)}", normal_style))
        
        # Build the PDF
        doc.build(elements)
        
        return filename
    
    except ImportError as e:
        console.print(f"[bold red]Error: Required PDF export libraries are missing. {str(e)}")
        console.print("[bold yellow]Try installing with: pip install reportlab")
        return ""
    except Exception as e:
        console.print(f"[bold red]Error exporting timeline as PDF: {str(e)}")
        return ""
