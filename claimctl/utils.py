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
            PageBreak,
            Image,
            KeepTogether,
            Flowable,
            CondPageBreak
        )
        from reportlab.graphics.shapes import Drawing, Line, Rect
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.linecharts import HorizontalLineChart
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen.canvas import Canvas

        # Create the Timeline_Exports directory if it doesn't exist
        exports_dir = Path("./Timeline_Exports")
        exports_dir.mkdir(exist_ok=True, parents=True)

        # Create a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(exports_dir / f"timeline_{matter_name}_{timestamp}.pdf")

        # Define sections for bookmarks and TOC
        sections = [
            {"name": "Executive Summary", "level": 0},
            {"name": "Financial Impact Summary", "level": 0},
            {"name": "Financial Impact by Event Type", "level": 1},
            {"name": "Monthly Financial Impact", "level": 1},
            {"name": "Financial Running Total Analysis", "level": 1},
            {"name": "Contradictions Analysis", "level": 0},
            {"name": "Timeline Events", "level": 0},
            {"name": "Chronological View", "level": 0},
            {"name": "Key Findings", "level": 0},
        ]

        # Custom flowable for section dividers
        class SectionDivider(Flowable):
            def __init__(self, width, height=0.1*inch, color=colors.lightblue):
                self.width = width
                self.height = height
                self.color = color
                Flowable.__init__(self)

            def draw(self):
                self.canv.setFillColor(self.color)
                self.canv.rect(0, 0, self.width, self.height, fill=True, stroke=False)

        # Custom flowable for horizontal timeline
        class HorizontalTimeline(Flowable):
            def __init__(self, events, width, height=2*inch):
                Flowable.__init__(self)
                self.events = events
                self.width = width
                self.height = height

            def draw(self):
                if not self.events:
                    return

                # Find date range
                dates = []
                for event in self.events:
                    if event.get("event_date"):
                        try:
                            date_obj = datetime.datetime.fromisoformat(event["event_date"]).date()
                            dates.append(date_obj)
                        except (ValueError, TypeError):
                            pass

                if not dates:
                    return

                min_date = min(dates)
                max_date = max(dates)
                date_range = (max_date - min_date).days
                if date_range == 0:
                    date_range = 30  # Prevent division by zero

                # Draw the horizontal line
                self.canv.setStrokeColor(colors.black)
                self.canv.setLineWidth(2)
                y_pos = self.height / 2
                self.canv.line(0, y_pos, self.width, y_pos)

                # Draw tick marks for every month
                current_date = datetime.date(min_date.year, min_date.month, 1)
                end_date = datetime.date(max_date.year, max_date.month, 1)

                while current_date <= end_date:
                    days_from_start = (current_date - min_date).days
                    x_pos = (days_from_start / date_range) * self.width

                    self.canv.line(x_pos, y_pos - 5, x_pos, y_pos + 5)

                    # Add month label
                    month_name = current_date.strftime("%b %Y")
                    self.canv.setFont("Helvetica", 7)
                    self.canv.drawCentredString(x_pos, y_pos - 15, month_name)

                    # Move to next month
                    if current_date.month == 12:
                        current_date = datetime.date(current_date.year + 1, 1, 1)
                    else:
                        current_date = datetime.date(current_date.year, current_date.month + 1, 1)

                # Plot events on the timeline
                self.canv.setFont("Helvetica", 6)

                # Group events by date to handle overlaps
                date_events = {}
                for event in self.events:
                    if event.get("event_date"):
                        try:
                            date_obj = datetime.datetime.fromisoformat(event["event_date"]).date()
                            if date_obj not in date_events:
                                date_events[date_obj] = []
                            date_events[date_obj].append(event)
                        except (ValueError, TypeError):
                            pass

                # Draw events
                for date_obj, events_on_date in date_events.items():
                    days_from_start = (date_obj - min_date).days
                    x_pos = (days_from_start / date_range) * self.width

                    # Set color based on event type
                    event_type = events_on_date[0].get("event_type", "other")
                    type_colors = {
                        "delay": colors.red,
                        "change_order": colors.blue,
                        "payment": colors.green,
                        "notice": colors.purple,
                        "communication": colors.lightgrey,
                    }
                    color = type_colors.get(event_type, colors.black)

                    # Draw marker (dot)
                    self.canv.setFillColor(color)
                    self.canv.circle(x_pos, y_pos, 4, fill=1)

                    # Draw event label with offset for multiple events
                    for i, event in enumerate(events_on_date):
                        offset = i * 15
                        self.canv.drawCentredString(
                            x_pos,
                            y_pos + 15 + offset,
                            f"{date_obj.day} - {event.get('event_type', '')}"
                        )

                        # Draw financial impact if available
                        if "financial_impact" in event and event["financial_impact"] is not None:
                            amount = event["financial_impact"]
                            if amount >= 0:
                                self.canv.setFillColor(colors.green)
                                impact_text = f"${amount:,.0f}"
                            else:
                                self.canv.setFillColor(colors.red)
                                impact_text = f"-${abs(amount):,.0f}"
                            self.canv.drawCentredString(x_pos, y_pos + 25 + offset, impact_text)
                            self.canv.setFillColor(color)

        # Document metadata and properties
        metadata = {
            "Title": f"Claim Timeline: {matter_name}",
            "Subject": "Claim Analysis Timeline Report",
            "Author": "Claim Assistant",
            "Creator": "Claim Assistant Timeline Generator",
            "Producer": "ReportLab PDF Library",
            "CreationDate": datetime.datetime.now(),
        }

        # Create the PDF document with metadata
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title=metadata["Title"],
            author=metadata["Author"],
            subject=metadata["Subject"],
            creator=metadata["Creator"],
            producer=metadata["Producer"],
        )

        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading1"]
        heading2_style = styles["Heading2"]
        heading3_style = styles["Heading3"]
        # Customize normal style for better wrapping
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            wordWrap='CJK',  # Better word wrapping
            allowWidows=0,
            allowOrphans=0
        )

        # Add special style for table cells to ensure proper wrapping
        cell_style = ParagraphStyle(
            'CellStyle',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            wordWrap='CJK',  # Better word wrapping
            allowWidows=0,
            allowOrphans=0
        )

        # Style for header cells
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            fontName='Helvetica-Bold',
            wordWrap='CJK',
            allowWidows=0,
            allowOrphans=0
        )

        # Create custom styles
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue
        )

        financial_positive_style = ParagraphStyle(
            'FinancialPositive',
            parent=styles['Normal'],
            textColor=colors.green
        )

        financial_negative_style = ParagraphStyle(
            'FinancialNegative',
            parent=styles['Normal'],
            textColor=colors.red
        )

        contradiction_style = ParagraphStyle(
            'Contradiction',
            parent=styles['Normal'],
            textColor=colors.red,
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.red,
            borderPadding=5
        )

        toc_heading_style = ParagraphStyle(
            'TOCHeading',
            parent=styles['Heading1'],
            fontSize=16
        )

        toc_entry_style = ParagraphStyle(
            'TOCEntry',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            leading=18
        )

        toc_entry_level1_style = ParagraphStyle(
            'TOCEntryLevel1',
            parent=toc_entry_style,
            leftIndent=20
        )

        key_finding_style = ParagraphStyle(
            'KeyFinding',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=18,
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.darkgrey,
            borderPadding=5,
            spaceAfter=12,
            wordWrap='CJK'  # Better word wrapping
        )

        # Create content elements list
        elements = []
        bookmarks = []  # For PDF bookmarks
        toc_entries = []  # For table of contents

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

            if filters.get("event_types") and len(filters["event_types"]) > 0:
                filter_parts.append(f"Event types: {', '.join(filters['event_types'])}")
            if filters.get("date_from"):
                filter_parts.append(f"From: {filters['date_from']}")
            if filters.get("date_to"):
                filter_parts.append(f"To: {filters['date_to']}")
            if filters.get("min_importance") is not None:
                filter_parts.append(f"Min importance: {filters['min_importance']}")
            if filters.get("min_confidence") is not None:
                filter_parts.append(f"Min confidence: {filters['min_confidence']}")

            if filter_parts:
                filter_text += "; ".join(filter_parts)
                elements.append(Paragraph(filter_text, normal_style))
                elements.append(Spacer(1, 0.25*inch))

        # Add table of contents placeholder - will be filled later
        toc_title = Paragraph("Table of Contents", toc_heading_style)
        elements.append(toc_title)
        elements.append(Spacer(1, 0.25*inch))

        # Add placeholder for TOC entries
        toc_placeholder = []  # Will be filled with actual TOC after first pass
        elements.extend(toc_placeholder)

        elements.append(PageBreak())

        # Add summary section
        summary_bookmark = len(elements)
        elements.append(Paragraph("Executive Summary", heading_style))
        elements.append(SectionDivider(6.5*inch))
        elements.append(Spacer(1, 0.25*inch))
        toc_entries.append(("Executive Summary", 0, summary_bookmark))

        # Get events for the horizontal timeline
        events = timeline_data.get("events", [])

        # Add horizontal timeline if events exist
        if events:
            elements.append(Paragraph("Project Timeline Overview", heading3_style))
            elements.append(Spacer(1, 0.05*inch))
            elements.append(HorizontalTimeline(events, 6.5*inch))
            elements.append(Spacer(1, 0.1*inch))

        # Add summary content
        summary = timeline_data.get("summary", "No timeline summary available.")

        # Split summary into paragraphs
        summary_paragraphs = summary.split('\n\n')
        for para in summary_paragraphs:
            if para.strip():
                elements.append(Paragraph(para.strip(), normal_style))
                elements.append(Spacer(1, 0.1*inch))

        elements.append(Spacer(1, 0.25*inch))
        elements.append(PageBreak())

        # Add financial summary if available
        financial_summary = timeline_data.get("financial_summary")
        if financial_summary:
            fin_summary_bookmark = len(elements)
            elements.append(Paragraph("Financial Impact Summary", heading_style))
            elements.append(SectionDivider(6.5*inch))
            elements.append(Spacer(1, 0.25*inch))
            toc_entries.append(("Financial Impact Summary", 0, fin_summary_bookmark))

            # Create financial summary table
            financial_data = [
                ["Metric", "Value"],
                ["Total Financial Impact", f"${financial_summary.get('total_amount', 0):,.2f}"],
                ["Positive Impacts", f"${financial_summary.get('total_positive', 0):,.2f}"],
                ["Negative Impacts", f"${financial_summary.get('total_negative', 0):,.2f}"],
                ["Financial Events", str(financial_summary.get('event_count', 0))],
            ]

            financial_table = Table(financial_data, colWidths=[2.5*inch, 2.5*inch])
            financial_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Colorize positive and negative impacts
                ('TEXTCOLOR', (1, 2), (1, 2), colors.green),  # Positive impacts
                ('TEXTCOLOR', (1, 3), (1, 3), colors.red),    # Negative impacts
            ]))

            elements.append(financial_table)
            elements.append(Spacer(1, 0.25*inch))

            # Add breakdown by event type if available
            if financial_summary.get('event_type_totals'):
                fin_by_type_bookmark = len(elements)
                elements.append(Paragraph("Financial Impact by Event Type", heading3_style))
                elements.append(Spacer(1, 0.15*inch))
                toc_entries.append(("Financial Impact by Event Type", 1, fin_by_type_bookmark))

                # Create data for event type breakdown
                event_type_data = [["Event Type", "Amount", "% of Total"]]
                total_amount = financial_summary.get('total_amount', 0)

                for event_type, amount in sorted(financial_summary['event_type_totals'].items(),
                                               key=lambda x: abs(x[1]), reverse=True):
                    percent = 0 if total_amount == 0 else (abs(amount) / abs(total_amount)) * 100
                    amount_str = f"${amount:,.2f}"
                    event_type_data.append([event_type, amount_str, f"{percent:.1f}%"])

                # Create the table
                event_type_table = Table(event_type_data, colWidths=[2*inch, 2*inch, 1.5*inch])
                event_type_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 1), (1, -1), 'RIGHT'),  # Right align amounts
                    ('ALIGN', (2, 1), (2, -1), 'RIGHT'),  # Right align percentages
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('WORDWRAP', (0, 0), (-1, -1), True),  # Enable wordwrap for all columns
                ]))

                elements.append(event_type_table)
                elements.append(Spacer(1, 0.25*inch))

                # Create a bar chart for financial impact by event type
                if len(financial_summary['event_type_totals']) > 1:
                    try:
                        drawing = Drawing(500, 250)
                        chart = VerticalBarChart()
                        chart.x = 50
                        chart.y = 50
                        chart.height = 150
                        chart.width = 375

                        # Extract data
                        data = []
                        categories = []
                        amounts = []

                        for event_type, amount in sorted(financial_summary['event_type_totals'].items(),
                                                      key=lambda x: abs(x[1]), reverse=True)[:5]:  # Limit to top 5
                            categories.append(event_type)
                            amounts.append(amount)

                        data.append(amounts)

                        chart.data = data
                        chart.categoryAxis.categoryNames = categories
                        chart.categoryAxis.labels.boxAnchor = 'ne'
                        chart.categoryAxis.labels.angle = 30
                        chart.categoryAxis.labels.dx = -8
                        chart.categoryAxis.labels.dy = -2

                        # Set colors
                        chart.bars[0].fillColor = colors.lightblue

                        # Disable value labels to avoid formatting errors
                        chart.barLabels = None

                        drawing.add(chart)
                        elements.append(drawing)
                        elements.append(Spacer(1, 0.25*inch))
                    except Exception as chart_error:
                        # Ignore chart errors - charts are optional
                        pass

                # Add explanation text
                elements.append(Paragraph("The chart above shows the financial impact breakdown by event type, helping to identify categories with the largest monetary effects on the project.", normal_style))
                elements.append(Spacer(1, 0.25*inch))

            # Add monthly financial chart if available
            if financial_summary.get('monthly_totals') and len(financial_summary['monthly_totals']) > 1:
                monthly_fin_bookmark = len(elements)
                elements.append(Paragraph("Monthly Financial Impact", heading3_style))
                elements.append(Spacer(1, 0.15*inch))
                toc_entries.append(("Monthly Financial Impact", 1, monthly_fin_bookmark))

                # Create data for monthly chart
                try:
                    monthly_data = [["Month", "Amount", "Running Total"]]
                    running_total = 0
                    cumulative_data = []

                    for month, amount in sorted(financial_summary['monthly_totals'].items()):
                        year, month_num = month.split('-')
                        month_name = datetime.date(int(year), int(month_num), 1).strftime("%b %Y")
                        running_total += amount
                        amount_str = f"${amount:,.2f}"
                        running_total_str = f"${running_total:,.2f}"
                        monthly_data.append([month_name, amount_str, running_total_str])
                        cumulative_data.append((month_name, running_total))

                    # Create the table
                    monthly_table = Table(monthly_data, colWidths=[1.5*inch, 2*inch, 2*inch])
                    monthly_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),  # Right align amounts
                        ('ALIGN', (2, 1), (2, -1), 'RIGHT'),  # Right align totals
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('WORDWRAP', (0, 0), (-1, -1), True),  # Enable wordwrap for all columns
                    ]))

                    elements.append(monthly_table)
                    elements.append(Spacer(1, 0.25*inch))

                    # Create a line chart for cumulative financial impact
                    if len(cumulative_data) > 1:
                        drawing = Drawing(500, 250)
                        chart = HorizontalLineChart()
                        chart.x = 50
                        chart.y = 50
                        chart.height = 150
                        chart.width = 375
                        # Chart title (removed because it's not supported in HorizontalLineChart)

                        # Extract data
                        categories = [d[0] for d in cumulative_data]
                        amounts = [d[1] for d in cumulative_data]

                        chart.data = [amounts]
                        chart.categoryAxis.categoryNames = categories
                        chart.categoryAxis.labels.boxAnchor = 'n'
                        chart.categoryAxis.labels.angle = 30
                        chart.categoryAxis.labels.dx = 0
                        chart.categoryAxis.labels.dy = -2

                        # Add lines and markers
                        chart.lines[0].strokeColor = colors.blue
                        chart.lines[0].strokeWidth = 2
                        # Symbol configuration removed due to compatibility issues
                        # chart.lines[0].symbol = 'FilledCircle'
                        # chart.lines[0].symbolSize = 5

                        drawing.add(chart)
                        elements.append(drawing)
                        elements.append(Spacer(1, 0.25*inch))

                        # Add explanation text
                        elements.append(Paragraph("The chart above shows the cumulative financial impact over time, illustrating how financial events compound throughout the project timeline.", normal_style))
                        elements.append(Spacer(1, 0.15*inch))
                except Exception as chart_error:
                    # Ignore chart errors - charts are optional
                    console.print(f"[yellow]Error generating monthly chart: {str(chart_error)}")

            # Add running total analysis
            fin_running_bookmark = len(elements)
            elements.append(Paragraph("Financial Running Total Analysis", heading3_style))
            elements.append(Spacer(1, 0.15*inch))
            toc_entries.append(("Financial Running Total Analysis", 1, fin_running_bookmark))

            elements.append(Paragraph("The financial events create a cumulative impact on the project finances. The running total represents the project's financial position at any given point in time, accounting for all financial events up to that date.", normal_style))
            elements.append(Spacer(1, 0.25*inch))

            # Extract financial events and sort by date
            financial_events = []
            for event in events:
                if "financial_impact" in event and event["financial_impact"] is not None:
                    financial_events.append(event)

            if financial_events:
                # Sort by date (with default for None values)
                financial_events.sort(key=lambda e: e.get("event_date", "") or "")

                # Calculate running totals
                running_total = 0
                running_total_data = [["Date", "Event Type", "Description", "Amount", "Running Total"]]

                for event in financial_events:
                    amount = event.get("financial_impact", 0)
                    running_total += amount

                    # Format the data
                    event_date = event.get("event_date", "Unknown")
                    event_type = event.get("event_type", "Other")
                    description = event.get("description", "")

                    # Create a Paragraph object for the description to enable proper wrapping
                    description_para = Paragraph(description, cell_style)

                    amount_str = f"${amount:,.2f}" if amount >= 0 else f"(${abs(amount):,.2f})"
                    total_str = f"${running_total:,.2f}" if running_total >= 0 else f"(${abs(running_total):,.2f})"

                    # Create paragraph objects for all cells
                    event_date_para = Paragraph(event_date, cell_style)
                    event_type_para = Paragraph(event_type, cell_style)
                    amount_str_para = Paragraph(amount_str, cell_style)
                    total_str_para = Paragraph(total_str, cell_style)

                    running_total_data.append([
                        event_date_para, event_type_para, description_para, amount_str_para, total_str_para
                    ])

                # Create running total table
                running_table = Table(running_total_data, colWidths=[0.9*inch, 0.9*inch, 2.2*inch, 1*inch, 1*inch])
                running_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (3, 1), (4, -1), 'RIGHT'),  # Right align amounts
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('VALIGN', (2, 1), (2, -1), 'TOP'),  # Top align descriptions
                    ('WORDWRAP', (0, 0), (-1, -1), True),  # Enable wordwrap for all columns
                ]))

                elements.append(running_table)
                elements.append(Spacer(1, 0.25*inch))

            elements.append(PageBreak())

        # Add contradiction section if available
        contradictions = timeline_data.get("contradictions", [])
        if contradictions:
            contradictions_bookmark = len(elements)
            elements.append(Paragraph("Contradictions Analysis", heading_style))
            elements.append(SectionDivider(6.5*inch))
            elements.append(Spacer(1, 0.25*inch))
            toc_entries.append(("Contradictions Analysis", 0, contradictions_bookmark))

            elements.append(Paragraph(f"The timeline contains {len(contradictions)} contradictions between events. These contradictions may indicate conflicting information or interpretations that require further investigation.", normal_style))
            elements.append(Spacer(1, 0.25*inch))

            # Create contradictions table
            contradiction_data = [[
                Paragraph("Type", header_style),
                Paragraph("Events", header_style),
                Paragraph("Details", header_style),
                Paragraph("Severity", header_style)
            ]]

            for i, contradiction in enumerate(contradictions):
                event1_date = contradiction.get("event1_date", "Unknown")
                event2_date = contradiction.get("event2_date", "Unknown")
                event_type = contradiction.get("event_type", "Unknown")
                details = contradiction.get("details", "")

                # Create a Paragraph object for details to enable proper wrapping
                details_para = Paragraph(details, cell_style)

                # Format the events dates - shorten to save space
                events_str = f"{event1_date} vs\n{event2_date}"

                # Estimate severity based on event type and content
                severity = "High" if "payment" in event_type or "delay" in event_type else "Medium"

                # Create paragraph objects for all columns
                event_type_para = Paragraph(event_type, cell_style)
                events_str_para = Paragraph(events_str, cell_style)
                severity_para = Paragraph(severity, cell_style)

                contradiction_data.append([
                    event_type_para,
                    events_str_para,
                    details_para,
                    severity_para
                ])

            # Create and style the table with better column widths
            contradiction_table = Table(contradiction_data, colWidths=[0.8*inch, 1.2*inch, 3.5*inch, 0.7*inch])
            contradiction_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightpink),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                # Color-code severity column
                ('TEXTCOLOR', (3, 1), (3, -1), colors.red),
                # Enable wordwrap for all columns
                ('WORDWRAP', (0, 0), (-1, -1), True),
            ]))

            elements.append(contradiction_table)

            # Add explanation for contradiction analysis
            elements.append(Spacer(1, 0.25*inch))
            elements.append(Paragraph("Contradictions indicate potential issues where different documents or communications present conflicting information. These should be carefully investigated and reconciled to determine the accurate project history.", normal_style))

            elements.append(Spacer(1, 0.25*inch))
            elements.append(PageBreak())

        # Add timeline events table
        events_bookmark = len(elements)
        elements.append(Paragraph("Timeline Events", heading_style))
        elements.append(SectionDivider(6.5*inch))
        elements.append(Spacer(1, 0.15*inch))
        toc_entries.append(("Timeline Events", 0, events_bookmark))

        if not events:
            elements.append(Paragraph("No timeline events found.", normal_style))
        else:
            # Add filtering and sorting options explanation
            elements.append(Paragraph("The timeline events below are presented in chronological order. Events with financial impacts or contradictions are highlighted.", normal_style))
            elements.append(Spacer(1, 0.2*inch))

            # Create event table with financial info and contradiction flags
            data = [[
                Paragraph("Date", header_style),
                Paragraph("Type", header_style),
                Paragraph("Description", header_style),
                Paragraph("Financial Impact", header_style),
                Paragraph("Flags", header_style),
                Paragraph("Document", header_style)
            ]]

            # Add each event to the table
            for event in events:
                event_date = event.get("event_date", "Unknown")
                event_type = event.get("event_type", "other")
                description = event.get("description", "")

                # Create a Paragraph object for description to enable proper wrapping
                description_para = Paragraph(description, cell_style)
                # Limit document name length to prevent overflow
                document_name = event.get("document", {}).get("file_name", "Unknown")
                if len(document_name) > 15:
                    document_name = document_name[:12] + "..."

                # Format financial impact - use shorter format to save space
                financial_impact = ""
                if "financial_impact" in event and event["financial_impact"] is not None:
                    amount = event["financial_impact"]
                    if amount >= 0:
                        # Use shorter format for large numbers (K for thousands, M for millions)
                        if abs(amount) >= 1000000:
                            financial_impact = f"${amount/1000000:.1f}M"
                        elif abs(amount) >= 1000:
                            financial_impact = f"${amount/1000:.1f}K"
                        else:
                            financial_impact = f"${amount:,.0f}"
                    else:
                        if abs(amount) >= 1000000:
                            financial_impact = f"-${abs(amount)/1000000:.1f}M"
                        elif abs(amount) >= 1000:
                            financial_impact = f"-${abs(amount)/1000:.1f}K"
                        else:
                            financial_impact = f"-${abs(amount):,.0f}"

                # Format contradiction flags
                flags = ""
                if event.get("has_contradiction", False):
                    flags = "⚠ Contradiction"

                # Create paragraph objects for all cells
                event_date_para = Paragraph(event_date, cell_style)
                event_type_para = Paragraph(event_type, cell_style)
                financial_impact_para = Paragraph(financial_impact, cell_style)
                flags_para = Paragraph(flags, cell_style)
                document_name_para = Paragraph(document_name, cell_style)

                data.append([
                    event_date_para,
                    event_type_para,
                    description_para,
                    financial_impact_para,
                    flags_para,
                    document_name_para
                ])

            # Create and style the table with wordwrap for the description column
            table = Table(data, repeatRows=1, colWidths=[0.8*inch, 0.8*inch, 2.4*inch, 0.9*inch, 0.8*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Allow wordwrap for all columns
                ('VALIGN', (2, 1), (2, -1), 'TOP'),
                ('WORDWRAP', (0, 0), (-1, -1), True),
                # Highlight contradiction flags
                ('TEXTCOLOR', (4, 1), (4, -1), colors.red),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 0.25*inch))
            elements.append(PageBreak())

        # Add chronological view
        chron_bookmark = len(elements)
        elements.append(Paragraph("Chronological View", heading_style))
        elements.append(SectionDivider(6.5*inch))
        elements.append(Spacer(1, 0.25*inch))
        toc_entries.append(("Chronological View", 0, chron_bookmark))

        # Group events by year and month
        events_by_month = timeline_data.get("events_by_month", {})
        if not events_by_month:
            elements.append(Paragraph("No timeline events found for chronological view.", normal_style))
        else:
            elements.append(Paragraph("This section presents the timeline events organized chronologically by year and month, showing the progression of events over time.", normal_style))
            elements.append(Spacer(1, 0.25*inch))

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

            # Use a simplified chronological view with a table instead
            chronological_data = [[
                Paragraph("Date", header_style),
                Paragraph("Type", header_style),
                Paragraph("Description", header_style),
                Paragraph("Financial Impact", header_style),
                Paragraph("Document", header_style)
            ]]

            # Sort all events by date
            all_events = []
            for year in years:
                for month in years[year]:
                    for event in years[year][month]:
                        all_events.append(event)

            # Sort events chronologically
            all_events.sort(key=lambda e: e.get("event_date", "") or "")

            # Add all events to the table
            for event in all_events:
                event_date = event.get("event_date", "Unknown")
                event_type = event.get("event_type", "other")
                description = event.get("description", "")

                # Create a Paragraph object for description to enable proper wrapping
                description_para = Paragraph(description, cell_style)
                # Limit document name length to prevent overflow
                document_name = event.get("document", {}).get("file_name", "Unknown")
                if len(document_name) > 15:
                    document_name = document_name[:12] + "..."

                # Format financial impact - use shorter format to save space
                financial_impact = ""
                if "financial_impact" in event and event["financial_impact"] is not None:
                    amount = event["financial_impact"]
                    if amount >= 0:
                        # Use shorter format for large numbers (K for thousands, M for millions)
                        if abs(amount) >= 1000000:
                            financial_impact = f"${amount/1000000:.1f}M"
                        elif abs(amount) >= 1000:
                            financial_impact = f"${amount/1000:.1f}K"
                        else:
                            financial_impact = f"${amount:,.0f}"
                    else:
                        if abs(amount) >= 1000000:
                            financial_impact = f"-${abs(amount)/1000000:.1f}M"
                        elif abs(amount) >= 1000:
                            financial_impact = f"-${abs(amount)/1000:.1f}K"
                        else:
                            financial_impact = f"-${abs(amount):,.0f}"

                # Add contradiction marker if present
                if event.get("has_contradiction", False):
                    description = f"⚠ {description}"

                # Create paragraph object for description
                description_para = Paragraph(description, cell_style)

                # Create paragraph objects for all cells
                event_date_para = Paragraph(event_date, cell_style)
                event_type_para = Paragraph(event_type, cell_style)
                financial_impact_para = Paragraph(financial_impact, cell_style)
                document_name_para = Paragraph(document_name, cell_style)

                chronological_data.append([
                    event_date_para,
                    event_type_para,
                    description_para,
                    financial_impact_para,
                    document_name_para
                ])

            # Create and style the table with adjusted column widths
            chrono_table = Table(chronological_data, repeatRows=1, colWidths=[0.9*inch, 0.8*inch, 2.9*inch, 0.9*inch, 0.9*inch])
            chrono_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Allow wordwrap for all columns
                ('VALIGN', (2, 1), (2, -1), 'TOP'),
                ('WORDWRAP', (0, 0), (-1, -1), True),
            ]))

            elements.append(chrono_table)
            elements.append(Spacer(1, 0.25*inch))

        # Skip the key findings section for now due to layout issues
        # We'll add a simplified version in future updates

        # Create a custom canvas to add bookmarks, headers, footers and other document-level features
# Add total count footer
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Total events: {len(events)}", normal_style))
        
        # If financial summary is available, add financial event count
        if financial_summary:
            elements.append(Paragraph(f"Financial events: {financial_summary.get('event_count', 0)}", normal_style))
        
        # If contradictions are available, add contradiction count
        if contradictions:
            elements.append(Paragraph(f"Contradictions: {len(contradictions)}", normal_style))
        class TimelineDocTemplate(SimpleDocTemplate):
            def __init__(self, filename, **kw):
                SimpleDocTemplate.__init__(self, filename, **kw)
                self.bookmarks = []
                self.toc_entries = []

            def afterFlowable(self, flowable):
                """Track the positions of headings for bookmarks and table of contents."""
                if isinstance(flowable, Paragraph):
                    text = flowable.getPlainText()
                    style = flowable.style.name

                    if style in ('Heading1', 'Title'):
                        key = text
                        level = 0

                        # Add a bookmark to the PDF
                        self.canv.bookmarkPage(key)
                        self.canv.addOutlineEntry(key, key, level, 0)

                        # Store the bookmark info for the table of contents
                        self.bookmarks.append((key, level, self.page))

                    elif style == 'Heading2':
                        key = text
                        level = 1

                        # Add a bookmark to the PDF
                        self.canv.bookmarkPage(key)
                        self.canv.addOutlineEntry(key, key, level, 0)

                        # Store the bookmark info for the table of contents
                        self.bookmarks.append((key, level, self.page))

        # Build the PDF in two passes to create table of contents
        # First pass: gather bookmarks and page numbers
        doc_with_toc = TimelineDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title=metadata["Title"],
            author=metadata["Author"],
            subject=metadata["Subject"],
            creator=metadata["Creator"],
            producer=metadata["Producer"],
        )

        # Generate the PDF with blank TOC initially
        doc_with_toc.multiBuild(elements)

        # Now create the real TOC entries from bookmarks
        toc_entries = []
        for section, level, page in doc_with_toc.bookmarks:
            if level == 0:
                style = toc_entry_style
            else:
                style = toc_entry_level1_style
                section = "  " + section

            # Create the TOC entry with a page number reference
            entry = Paragraph(
                f"{section}{'.' * (50 - len(section))} {page}",
                style
            )
            toc_entries.append(entry)
            toc_entries.append(Spacer(1, 0.1*inch))

        # Replace the TOC placeholder with real entries
        elements[3:3+len(toc_placeholder)] = toc_entries

        # Second pass: build with TOC entries
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title=metadata["Title"],
            author=metadata["Author"],
            subject=metadata["Subject"],
            creator=metadata["Creator"],
            producer=metadata["Producer"],
        )

        # Define page template for headers and footers
        def header_footer(canvas, doc):
            # Save state for later restoration
            canvas.saveState()

            # Add page number to each page
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.setFont("Helvetica", 9)
            canvas.drawRightString(7.5*inch, 0.25*inch, text)

            # Add header with matter name
            if page_num > 1:  # Skip header on first page (title page)
                canvas.setFont("Helvetica-Bold", 10)
                canvas.drawString(0.5*inch, 10.5*inch, f"Claim Timeline: {matter_name}")
                canvas.setFont("Helvetica", 8)
                canvas.drawRightString(7.5*inch, 10.5*inch, f"Generated: {date_str}")

                # Add horizontal line under the header
                canvas.setStrokeColor(colors.lightgrey)
                canvas.line(0.5*inch, 10.4*inch, 7.5*inch, 10.4*inch)

            # Restore state
            canvas.restoreState()

        # Build the final document with headers and footers
        doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

        return filename

    except ImportError as e:
        console.print(f"[bold red]Error: Required PDF export libraries are missing. {str(e)}")
        console.print("[bold yellow]Try installing with: pip install reportlab")
        return ""
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        console.print(f"[bold red]Error exporting timeline as PDF: {str(e)}")
        console.print(f"[bold red]Traceback: {error_details}")
        return ""
