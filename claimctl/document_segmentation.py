"""
Document segmentation module for the claim-assistant system.

This module provides functionality to detect document boundaries in large PDFs
containing multiple logical documents. It uses a machine learning approach with
embeddings to identify semantic shifts that indicate document boundaries.
"""

import os
import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import fitz  # PyMuPDF

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

def create_text_segments(text: str, segment_size: int = 500, segment_stride: int = 100) -> List[Tuple[str, int]]:
    """
    Create overlapping text segments from a document.
    
    Args:
        text: Full text of the document
        segment_size: Size of each text segment
        segment_stride: Stride between consecutive segments
        
    Returns:
        List of tuples containing (segment_text, start_position)
    """
    segments = []
    
    # Handle case where text is shorter than segment_size
    if len(text) <= segment_size:
        return [(text, 0)]
    
    pos = 0
    while pos < len(text):
        end_pos = min(pos + segment_size, len(text))
        segment = text[pos:end_pos]
        segments.append((segment, pos))
        
        # Move position by stride
        pos += segment_stride
        
        # If we're near the end, make sure to include the final segment
        if pos < len(text) and pos + segment_size >= len(text):
            segments.append((text[pos:], pos))
            break
            
    return segments

def get_embeddings(text_segments: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Generate embeddings for text segments using the configured model.
    This is a placeholder function; in practice, you would use your existing
    embedding functionality from the claim-assistant system.
    
    Args:
        text_segments: List of text segments to embed
        config: Configuration dictionary with embedding model settings
        
    Returns:
        Array of embeddings for each segment
    """
    # In a real implementation, you would call your existing embedding code
    # For example:
    # from claimctl.search import get_embeddings
    # return get_embeddings(text_segments)
    
    # Placeholder implementation - replace with actual embedding generation
    # This simulates embedding generation for demonstration purposes
    try:
        # Try to import and use your existing embedding function
        from claimctl.search import get_embeddings as system_embeddings
        return system_embeddings(text_segments)
    except ImportError:
        # Fallback to a very simple mock embedding function for demonstration
        # This should be replaced with your actual embedding code
        logger.warning("Using mock embeddings. Replace with actual embedding function.")
        dim = 384  # Typical embedding dimension
        return np.random.rand(len(text_segments), dim)

def calculate_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between adjacent embeddings.
    
    Args:
        embeddings: Array of embedding vectors
        
    Returns:
        Array of similarity scores between adjacent embeddings
    """
    similarities = []
    
    for i in range(1, len(embeddings)):
        # Calculate cosine similarity between adjacent embeddings
        dot_product = np.dot(embeddings[i-1], embeddings[i])
        norm_a = np.linalg.norm(embeddings[i-1])
        norm_b = np.linalg.norm(embeddings[i])
        
        if norm_a == 0 or norm_b == 0:
            similarity = 0
        else:
            similarity = dot_product / (norm_a * norm_b)
            
        similarities.append(similarity)
        
    return np.array(similarities)

def find_potential_boundaries(similarities: np.ndarray, 
                             threshold_multiplier: float = 1.5) -> List[int]:
    """
    Find potential document boundaries based on drops in similarity.
    
    Args:
        similarities: Array of similarity scores
        threshold_multiplier: Controls sensitivity of boundary detection
        
    Returns:
        List of indices where potential boundaries occur
    """
    # Calculate statistics of similarities
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    # Calculate dynamic threshold based on statistics
    # Lower similarities indicate bigger semantic shifts
    threshold = mean_sim - (threshold_multiplier * std_sim)
    
    # Find segments where similarity drops below threshold
    potential_boundaries = [i for i, sim in enumerate(similarities) if sim < threshold]
    
    return potential_boundaries

def find_natural_break(text: str, position: int, window: int = 100) -> int:
    """
    Find a natural break point in text near the specified position.
    
    Args:
        text: The document text
        position: Approximate position of the boundary
        window: Window size around position to look for natural breaks
        
    Returns:
        Position of the natural break
    """
    start = max(0, position - window)
    end = min(len(text), position + window)
    
    segment = text[start:end]
    
    # Try to find paragraph breaks or page breaks
    paragraph_breaks = [m.start() + start for m in re.finditer(r'\n\s*\n', segment)]
    page_breaks = [m.start() + start for m in re.finditer(r'\f', segment)]
    
    # If we found paragraph or page breaks, use the closest one
    breaks = paragraph_breaks + page_breaks
    if breaks:
        # Find the break closest to the original position
        return min(breaks, key=lambda x: abs(x - position))
    
    # If no paragraph or page breaks, look for end of sentences
    sentence_breaks = [m.start() + start for m in re.finditer(r'[.!?]\s+[A-Z]', segment)]
    if sentence_breaks:
        return min(sentence_breaks, key=lambda x: abs(x - position))
    
    # Fall back to the original position if no better break is found
    return position

def score_boundary(text: str, position: int, similarity_score: float) -> float:
    """
    Score a boundary based on multiple factors.
    
    Args:
        text: The document text
        position: Position of the boundary
        similarity_score: Similarity score at the boundary
        
    Returns:
        Confidence score for the boundary (0-1)
    """
    # Start with inverted similarity as base score (lower similarity = higher score)
    base_score = 1 - similarity_score
    
    # Look for indicators of document boundaries
    context = text[max(0, position-200):min(len(text), position+200)]
    
    # Check for strong boundary indicators
    indicators = [
        r'(?i)(end\s+of\s+document)',
        r'(?i)(page\s+\d+\s+of\s+\d+$)',
        r'(?i)(exhibit\s+[a-z])',
        r'(?i)(appendix\s+[a-z])',
        r'(?i)(attachment\s+[a-z\d])',
        r'(?i)^\s*(date:)',
        r'(?i)^\s*(to:)',
        r'(?i)^\s*(from:)',
        r'(?i)^\s*(subject:)',
        r'(?i)(daily\s+report)',
        r'(?i)(meeting\s+minutes)',
        r'(?i)(change\s+order)'
    ]
    
    indicator_bonus = 0
    for pattern in indicators:
        if re.search(pattern, context):
            indicator_bonus += 0.1  # Add bonus for each indicator found
    
    # Cap indicator bonus at 0.5
    indicator_bonus = min(0.5, indicator_bonus)
    
    # Combine scores
    final_score = min(0.95, base_score + indicator_bonus)
    
    return final_score

def refine_boundaries(text: str, 
                     potential_boundaries: List[int], 
                     segment_positions: List[int],
                     similarities: List[float],
                     min_boundary_distance: int = 1000,
                     min_confidence: float = 0.3) -> List[Dict[str, Any]]:
    """
    Refine boundaries by finding natural breaks and filtering.
    
    Args:
        text: The document text
        potential_boundaries: List of potential boundary indices
        segment_positions: List of segment start positions
        similarities: List of similarity scores
        min_boundary_distance: Minimum distance between boundaries
        min_confidence: Minimum confidence score for a boundary
        
    Returns:
        List of refined boundary dictionaries
    """
    if not potential_boundaries:
        return []
        
    # Convert boundary indices to positions in the text
    boundary_positions = [segment_positions[i+1] for i in potential_boundaries]
    
    # Find natural breaks near each boundary position
    natural_positions = [find_natural_break(text, pos) for pos in boundary_positions]
    
    # Score boundaries
    boundaries = []
    for i, pos in enumerate(natural_positions):
        similarity = similarities[potential_boundaries[i]]
        confidence = score_boundary(text, pos, similarity)
        
        if confidence >= min_confidence:
            boundaries.append({
                'position': pos,
                'confidence': confidence,
                'original_index': potential_boundaries[i]
            })
    
    # Sort boundaries by position
    boundaries.sort(key=lambda x: x['position'])
    
    # Filter boundaries that are too close to each other
    filtered_boundaries = []
    last_position = -min_boundary_distance
    
    for boundary in boundaries:
        if boundary['position'] - last_position >= min_boundary_distance:
            filtered_boundaries.append(boundary)
            last_position = boundary['position']
    
    return filtered_boundaries

def visualize_boundaries(text: str, 
                        segment_positions: List[int], 
                        similarities: List[float], 
                        boundaries: List[Dict[str, Any]],
                        output_path: Optional[str] = None) -> None:
    """
    Generate visualization of document boundaries.
    
    Args:
        text: The document text
        segment_positions: List of segment positions
        similarities: List of similarity scores
        boundaries: List of boundary dictionaries
        output_path: Path to save visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping visualization.")
        return
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot similarity scores
    plt.subplot(2, 1, 1)
    plt.plot(similarities, label='Similarity')
    
    # Plot mean and threshold
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    threshold = mean_sim - (1.5 * std_sim)
    
    plt.axhline(y=mean_sim, color='g', linestyle='-', label='Mean')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    # Mark detected boundaries
    boundary_indices = [b['original_index'] for b in boundaries]
    boundary_sims = [similarities[i] for i in boundary_indices]
    plt.scatter(boundary_indices, boundary_sims, color='red', s=100, marker='o', label='Boundaries')
    
    plt.title('Document Boundary Detection')
    plt.ylabel('Similarity Score')
    plt.xlabel('Segment Index')
    plt.legend()
    
    # Plot confidence scores
    plt.subplot(2, 1, 2)
    boundary_positions = [b['position'] for b in boundaries]
    confidence_scores = [b['confidence'] for b in boundaries]
    
    plt.bar(range(len(boundaries)), confidence_scores)
    plt.xticks(range(len(boundaries)), [f"{pos//1000}K" for pos in boundary_positions])
    plt.title('Boundary Confidence Scores')
    plt.ylabel('Confidence')
    plt.xlabel('Text Position (K chars)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()

def detect_document_boundaries(text: str, 
                              config: Dict[str, Any] = None,
                              segment_size: int = 500, 
                              segment_stride: int = 100, 
                              threshold_multiplier: float = 1.5,
                              min_confidence: float = 0.3,
                              min_document_length: int = 1000,
                              min_boundary_distance: int = 2000,
                              visualize: bool = False,
                              visualization_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main function to detect document boundaries using ML approach with embeddings.
    
    Args:
        text: The document text
        config: Configuration dictionary
        segment_size: Size of text segments for analysis
        segment_stride: Stride between consecutive segments
        threshold_multiplier: Controls sensitivity of boundary detection
        min_confidence: Minimum confidence score for a boundary
        min_document_length: Minimum length of a document in characters
        min_boundary_distance: Minimum distance between boundaries
        visualize: Whether to generate visualization
        visualization_path: Path to save visualization
        
    Returns:
        List of boundary dictionaries with positions and confidence scores
    """
    if not text:
        logger.warning("Empty text provided to detect_document_boundaries")
        return []
        
    # Use provided config or create an empty one
    if config is None:
        config = {}
    
    # Step 1: Create overlapping text segments
    logger.info(f"Creating text segments with size={segment_size}, stride={segment_stride}")
    segments_with_pos = create_text_segments(text, segment_size, segment_stride)
    segments = [s[0] for s in segments_with_pos]
    segment_positions = [s[1] for s in segments_with_pos]
    
    # Step 2: Generate embeddings for segments
    logger.info(f"Generating embeddings for {len(segments)} segments")
    embeddings = get_embeddings(segments, config)
    
    # Step 3: Calculate similarity between adjacent segments
    logger.info("Calculating similarities between adjacent segments")
    similarities = calculate_similarities(embeddings)
    
    # Step 4: Find potential boundaries based on similarity drops
    logger.info(f"Finding potential boundaries with threshold multiplier={threshold_multiplier}")
    potential_boundaries = find_potential_boundaries(similarities, threshold_multiplier)
    
    # Step 5: Refine boundaries to natural breaks and filter
    logger.info(f"Refining boundaries with min_confidence={min_confidence}")
    boundaries = refine_boundaries(
        text, 
        potential_boundaries, 
        segment_positions,
        similarities,
        min_boundary_distance,
        min_confidence
    )
    
    # Step 6: Ensure document size constraints
    final_boundaries = []
    last_pos = 0
    
    for i, boundary in enumerate(boundaries):
        doc_length = boundary['position'] - last_pos
        
        # Skip boundaries that would create documents that are too small
        if doc_length >= min_document_length:
            final_boundaries.append(boundary)
            last_pos = boundary['position']
    
    # Step 7: Generate visualization if requested
    if visualize and MATPLOTLIB_AVAILABLE:
        logger.info("Generating boundary visualization")
        visualize_boundaries(
            text,
            segment_positions,
            similarities,
            final_boundaries,
            visualization_path
        )
    
    logger.info(f"Found {len(final_boundaries)} document boundaries")
    return final_boundaries

def map_text_positions_to_pages(pdf_path: str, text_positions: List[int]) -> List[int]:
    """
    Map text positions to PDF page numbers.
    
    Args:
        pdf_path: Path to the PDF file
        text_positions: List of character positions in the extracted text
        
    Returns:
        List of page numbers corresponding to the text positions
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    # Extract text with character positions from each page
    cumulative_length = 0
    page_boundaries = [0]  # Start with position 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        cumulative_length += len(text)
        page_boundaries.append(cumulative_length)
    
    # Map each text position to a page number
    for pos in text_positions:
        # Find the first page boundary that exceeds the position
        for i in range(1, len(page_boundaries)):
            if page_boundaries[i] > pos:
                pages.append(i - 1)  # Previous page contains this position
                break
        else:
            # If we don't find a page, use the last page
            pages.append(len(doc) - 1)
    
    doc.close()
    return pages

def split_pdf_by_boundaries(pdf_path: str, 
                           boundaries: List[Dict[str, Any]],
                           output_dir: str,
                           classify_documents: bool = True) -> List[Dict[str, Any]]:
    """
    Split a PDF file based on detected document boundaries.
    
    Args:
        pdf_path: Path to the PDF file
        boundaries: List of boundary dictionaries
        output_dir: Directory to save split PDFs
        classify_documents: Whether to classify document types
        
    Returns:
        List of metadata for the split documents
    """
    if not boundaries:
        logger.warning("No boundaries provided to split_pdf_by_boundaries")
        return []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text from PDF for classification and position mapping
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # Map text positions to page numbers
    boundary_positions = [b['position'] for b in boundaries]
    boundary_pages = map_text_positions_to_pages(pdf_path, boundary_positions)
    
    # Add start of document as first boundary
    document_ranges = [(0, boundary_pages[0])]
    
    # Add intermediate documents
    for i in range(len(boundary_pages) - 1):
        document_ranges.append((boundary_pages[i] + 1, boundary_pages[i + 1]))
    
    # Add last document
    document_ranges.append((boundary_pages[-1] + 1, len(doc) - 1))
    
    # Create output documents
    split_documents = []
    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
    
    for i, (start_page, end_page) in enumerate(document_ranges):
        # Skip invalid page ranges
        if start_page > end_page:
            continue
            
        # Create a new PDF
        output_pdf = fitz.open()
        
        # Copy pages from the original PDF
        for page_num in range(start_page, end_page + 1):
            output_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        # Determine document text for classification
        doc_start = 0 if i == 0 else boundaries[i-1]['position']
        doc_end = boundaries[i]['position'] if i < len(boundaries) else len(full_text)
        doc_text = full_text[doc_start:doc_end]
        
        # Classify document if requested
        doc_type = "unknown"
        confidence = 0.0
        
        if classify_documents and doc_text:
            try:
                # If document classification functionality exists, use it
                from claimctl.ingest import classify_document as system_classify
                doc_type, confidence = system_classify(doc_text)
            except ImportError:
                # Fallback to basic classification
                doc_type, confidence = classify_document(doc_text)
        
        # Save the output PDF
        output_path = os.path.join(output_dir, f"{pdf_name}_doc{i+1}_{doc_type}.pdf")
        output_pdf.save(output_path)
        output_pdf.close()
        
        # Add metadata
        split_documents.append({
            'path': output_path,
            'start_page': start_page,
            'end_page': end_page,
            'page_count': end_page - start_page + 1,
            'doc_type': doc_type,
            'confidence': confidence
        })
        
    doc.close()
    return split_documents

def classify_document(doc_text: str, doc_length: Optional[int] = None) -> Tuple[str, float]:
    """
    Basic document classification based on text content.
    This is a fallback if the system's document classification isn't available.
    
    Args:
        doc_text: Text content of the document
        doc_length: Length of the document in characters
        
    Returns:
        Tuple of (document_type, confidence_score)
    """
    if doc_length is None:
        doc_length = len(doc_text)
    
    # Extract a sample of the text for classification
    # Use beginning, some from middle, and end
    sample_size = min(2000, doc_length // 3)
    beginning = doc_text[:sample_size]
    middle_start = max(0, (doc_length // 2) - (sample_size // 2))
    middle = doc_text[middle_start:middle_start + sample_size]
    end = doc_text[max(0, doc_length - sample_size):]
    
    sample = beginning + middle + end
    
    # Simple pattern-based classification
    patterns = {
        'email': [r'(?i)from:', r'(?i)to:', r'(?i)subject:', r'(?i)sent:', r'@'],
        'invoice': [r'(?i)invoice', r'(?i)bill', r'(?i)payment', r'(?i)amount due', r'(?i)total:'],
        'change_order': [r'(?i)change order', r'(?i)change directive', r'(?i)modification', r'(?i)amendment'],
        'meeting_minutes': [r'(?i)meeting minutes', r'(?i)attendees:', r'(?i)present:', r'(?i)discussed:'],
        'daily_report': [r'(?i)daily report', r'(?i)daily log', r'(?i)work performed', r'(?i)weather:'],
        'rfi': [r'(?i)request for information', r'(?i)rfi', r'(?i)information requested'],
        'contract': [r'(?i)agreement', r'(?i)contract', r'(?i)terms and conditions', r'(?i)scope of work'],
        'schedule': [r'(?i)schedule', r'(?i)timeline', r'(?i)milestone', r'(?i)gantt'],
        'drawing': [r'(?i)drawing', r'(?i)figure', r'(?i)detail', r'(?i)section']
    }
    
    # Score each document type
    scores = {}
    
    for doc_type, patterns_list in patterns.items():
        score = 0
        for pattern in patterns_list:
            matches = re.findall(pattern, sample)
            score += len(matches)
        
        # Normalize score
        scores[doc_type] = min(1.0, score / (2 * len(patterns_list)))
    
    # Select highest scoring type
    if not scores or max(scores.values()) < 0.2:
        return 'unknown', 0.1
    
    best_type = max(scores.items(), key=lambda x: x[1])
    return best_type[0], best_type[1]

def process_pdf_for_segmentation(pdf_path: str, 
                                output_dir: str, 
                                config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process a PDF file to detect and segment logical documents within it.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save results
        config: Configuration dictionary
        
    Returns:
        Dictionary with results of the segmentation process
    """
    if config is None:
        config = {}
        
    # Get configuration values
    segment_size = config.get('SEGMENT_SIZE', 500)
    segment_stride = config.get('SEGMENT_STRIDE', 100)
    threshold_multiplier = config.get('THRESHOLD_MULTIPLIER', 1.5)
    min_confidence = config.get('MIN_CONFIDENCE', 0.3)
    min_document_length = config.get('MIN_DOCUMENT_LENGTH', 1000)
    min_boundary_distance = config.get('MIN_BOUNDARY_DISTANCE', 2000)
    visualize = config.get('VISUALIZE', False)
    
    # Extract text from PDF
    logger.info(f"Extracting text from {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # Create visualization path if needed
    visualization_path = None
    if visualize:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        visualization_path = os.path.join(vis_dir, f"{pdf_name}_boundaries.png")
    
    # Detect document boundaries
    logger.info(f"Detecting document boundaries in {pdf_path}")
    boundaries = detect_document_boundaries(
        text,
        config,
        segment_size=segment_size,
        segment_stride=segment_stride,
        threshold_multiplier=threshold_multiplier,
        min_confidence=min_confidence,
        min_document_length=min_document_length,
        min_boundary_distance=min_boundary_distance,
        visualize=visualize,
        visualization_path=visualization_path
    )
    
    # Split PDF based on detected boundaries
    logger.info(f"Splitting PDF into {len(boundaries)} documents")
    split_docs = split_pdf_by_boundaries(
        pdf_path,
        boundaries,
        output_dir,
        classify_documents=True
    )
    
    return {
        'original_pdf': pdf_path,
        'text_length': len(text),
        'documents_found': len(split_docs),
        'documents': split_docs,
        'visualization': visualization_path if visualize else None
    }