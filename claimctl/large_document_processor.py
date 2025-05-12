"""
Large document processor for efficient handling of very large PDFs.

This module provides optimized processing for very large PDFs (hundreds of pages)
with memory-efficient streaming, parallel processing, and robust error handling.
"""

import os
import gc
import time
import logging
from typing import List, Dict, Tuple, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class LargeDocumentConfig:
    """Configuration for large document processing."""
    
    # Size thresholds
    large_doc_threshold: int = 500000  # Characters
    very_large_doc_threshold: int = 5000000  # Characters
    
    # Segment settings
    optimal_segment_size: int = 100000  # Characters per segment
    segment_overlap: int = 10000  # Characters of overlap between segments
    
    # Processing settings
    max_workers: int = max(2, min(8, multiprocessing.cpu_count() // 2))
    batch_size: int = 25  # Chunks per batch
    timeout: int = 300  # Seconds
    
    # Memory management
    force_gc: bool = True  # Force garbage collection after processing segments
    
    # Duplicate detection
    similarity_threshold: float = 0.8  # Threshold for duplicate detection
    comparison_sample_size: int = 500  # Characters to sample for comparison


def is_large_document(text: str, config: Optional[LargeDocumentConfig] = None) -> bool:
    """
    Check if a document is considered large.
    
    Args:
        text: Document text
        config: Configuration (optional)
        
    Returns:
        True if document exceeds large_doc_threshold
    """
    config = config or LargeDocumentConfig()
    return len(text) > config.large_doc_threshold


def is_very_large_document(text: str, config: Optional[LargeDocumentConfig] = None) -> bool:
    """
    Check if a document is considered very large.
    
    Args:
        text: Document text
        config: Configuration (optional)
        
    Returns:
        True if document exceeds very_large_doc_threshold
    """
    config = config or LargeDocumentConfig()
    return len(text) > config.very_large_doc_threshold


def create_segments(text: str, config: Optional[LargeDocumentConfig] = None) -> List[Tuple[str, int]]:
    """
    Split document into overlapping segments for parallel processing.
    
    Args:
        text: Document text
        config: Configuration (optional)
        
    Returns:
        List of (segment_text, start_position) tuples
    """
    config = config or LargeDocumentConfig()
    
    # Calculate optimal segment size based on document size
    segment_size = config.optimal_segment_size
    overlap = config.segment_overlap
    
    # Adjust segment size for very large documents
    if is_very_large_document(text, config):
        # Scale down segment size for extremely large documents
        segment_size = min(segment_size, len(text) // (10 * config.max_workers))
        segment_size = max(segment_size, 50000)  # Ensure minimum segment size
    
    segments = []
    pos = 0
    
    while pos < len(text):
        # Find a good break point for the end of this segment
        end = min(pos + segment_size, len(text))
        
        # If not at the end of text, try to find a clean break point
        if end < len(text):
            # Look for paragraph break
            next_break = text.find('\n\n', end - 200, end + 200)
            if next_break > 0:
                end = next_break + 2  # Include the double newline
            else:
                # Look for single newline
                next_break = text.find('\n', end - 100, end + 100)
                if next_break > 0:
                    end = next_break + 1  # Include the newline
                else:
                    # Look for sentence end
                    for i in range(end - 50, min(end + 50, len(text))):
                        if i > 0 and text[i-1:i+1] in ['. ', '? ', '! ']:
                            end = i + 1
                            break
        
        # Create segment
        segment = text[pos:end]
        segments.append((segment, pos))
        
        # Move position for next segment with overlap
        pos = end - overlap if end < len(text) else len(text)
    
    return segments


def process_segment(segment: Tuple[str, int], 
                   process_func: Callable, 
                   process_args: Dict[str, Any]) -> Tuple[List[Any], int, int]:
    """
    Process a document segment.
    
    Args:
        segment: (segment_text, start_position) tuple
        process_func: Function to process the segment
        process_args: Arguments for the process function
        
    Returns:
        Tuple of (processing_results, start_position, end_position)
    """
    segment_text, start_position = segment
    
    # Process this segment
    results = process_func(segment_text, **process_args)
    
    # Return results with position information
    return results, start_position, start_position + len(segment_text)


def is_duplicate_chunk(chunk1: Dict[str, Any], chunk2: Dict[str, Any], 
                       config: Optional[LargeDocumentConfig] = None) -> bool:
    """
    Check if two chunks are duplicates or near-duplicates.
    
    Args:
        chunk1: First chunk
        chunk2: Second chunk
        config: Configuration (optional)
        
    Returns:
        True if chunks are considered duplicates
    """
    config = config or LargeDocumentConfig()
    
    # Get text from chunks
    text1 = chunk1.get('text', '')
    text2 = chunk2.get('text', '')
    
    # Simple length check first
    if abs(len(text1) - len(text2)) / max(1, max(len(text1), len(text2))) > 0.2:
        return False
    
    # Sample from beginning, middle, and end for efficiency
    sample_size = min(config.comparison_sample_size, 
                      min(len(text1), len(text2)) // 3)
    
    if sample_size == 0:
        return False
    
    # Check beginning
    beginning_similarity = jaccard_similarity(
        text1[:sample_size], 
        text2[:sample_size]
    )
    
    # Check middle
    mid1 = max(0, len(text1) // 2 - sample_size // 2)
    mid2 = max(0, len(text2) // 2 - sample_size // 2)
    middle_similarity = jaccard_similarity(
        text1[mid1:mid1 + sample_size],
        text2[mid2:mid2 + sample_size]
    )
    
    # Check end
    end_similarity = jaccard_similarity(
        text1[-sample_size:] if len(text1) >= sample_size else text1,
        text2[-sample_size:] if len(text2) >= sample_size else text2
    )
    
    # Calculate overall similarity
    overall_similarity = (beginning_similarity + middle_similarity + end_similarity) / 3
    
    return overall_similarity > config.similarity_threshold


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two text strings.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0-1.0)
    """
    # Convert to sets of characters for efficient comparison
    set1 = set(text1)
    set2 = set(text2)
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / max(1, union)


def remove_duplicate_chunks(chunks: List[Dict[str, Any]], 
                           config: Optional[LargeDocumentConfig] = None) -> List[Dict[str, Any]]:
    """
    Remove duplicate chunks from segment overlaps.
    
    Args:
        chunks: List of chunk dictionaries
        config: Configuration (optional)
        
    Returns:
        List of unique chunks
    """
    if not chunks:
        return []
    
    config = config or LargeDocumentConfig()
    unique_chunks = []
    
    # First chunk is always unique
    unique_chunks.append(chunks[0])
    
    # Compare each chunk with previously kept chunks
    for chunk in chunks[1:]:
        is_duplicate = False
        
        # Check last few unique chunks for duplicates (more efficient than checking all)
        for unique_chunk in unique_chunks[-5:]:
            if is_duplicate_chunk(chunk, unique_chunk, config):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    return unique_chunks


def process_large_document(text: str, 
                          process_func: Callable,
                          process_args: Dict[str, Any] = None,
                          config: Optional[LargeDocumentConfig] = None,
                          progress_callback: Optional[Callable[[float], None]] = None) -> List[Any]:
    """
    Process a large document in parallel segments with memory optimization.
    
    Args:
        text: Document text
        process_func: Function to process each segment
        process_args: Arguments for the process function
        config: Configuration (optional)
        progress_callback: Function to call with progress updates
        
    Returns:
        Combined processing results
    """
    if not text:
        return []
    
    config = config or LargeDocumentConfig()
    process_args = process_args or {}
    
    # Check if the document is large enough to need optimized processing
    if not is_large_document(text, config):
        # For small documents, just process directly
        return process_func(text, **process_args)
    
    # Log the start of large document processing
    logger.info(f"Starting optimized processing for large document ({len(text)} chars)")
    start_time = time.time()
    
    # Split document into segments
    segments = create_segments(text, config)
    logger.info(f"Document split into {len(segments)} segments for parallel processing")
    
    # Process segments in parallel
    all_results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_segment, segment, process_func, process_args): i 
            for i, segment in enumerate(segments)
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                result, start_pos, end_pos = future.result()
                all_results.append((result, start_pos, end_pos))
                
                # Call progress callback
                completed += 1
                if progress_callback:
                    progress_callback(completed / len(segments))
                
            except Exception as e:
                logger.error(f"Error processing segment: {e}")
                
            # Force garbage collection if configured
            if config.force_gc:
                gc.collect()
    
    # Sort results by segment start position
    all_results.sort(key=lambda x: x[1])
    
    # Combine and deduplicate results
    combined_results = []
    for result, _, _ in all_results:
        combined_results.extend(result)
    
    # Remove duplicates from segment overlaps
    unique_results = remove_duplicate_chunks(combined_results, config)
    
    # Log completion
    elapsed = time.time() - start_time
    logger.info(f"Large document processing completed in {elapsed:.2f}s. "
                f"Found {len(unique_results)} unique chunks from {len(combined_results)} total.")
    
    return unique_results


def streaming_document_generator(document_path: str, 
                                chunk_size: int = 1000000) -> Iterator[str]:
    """
    Generator that streams a document in chunks to reduce memory usage.
    
    Args:
        document_path: Path to document file
        chunk_size: Size of each chunk in bytes
        
    Yields:
        Document text chunks
    """
    with open(document_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def estimate_memory_requirements(text_length: int) -> Dict[str, float]:
    """
    Estimate memory requirements for processing a document.
    
    Args:
        text_length: Length of document text
        
    Returns:
        Dictionary of memory estimates in MB
    """
    # Approximate memory usage based on empirical measurements
    text_memory = text_length * 2 / 1024 / 1024  # Text in MB
    embedding_memory = text_length * 0.5 / 1024 / 1024  # Embeddings in MB
    processing_memory = text_length * 4 / 1024 / 1024  # Processing overhead in MB
    
    return {
        'text': text_memory,
        'embeddings': embedding_memory,
        'processing': processing_memory,
        'total': text_memory + embedding_memory + processing_memory
    }


def adjust_processing_parameters(text_length: int, 
                                available_memory: Optional[int] = None) -> LargeDocumentConfig:
    """
    Dynamically adjust processing parameters based on document size.
    
    Args:
        text_length: Length of document text
        available_memory: Available memory in MB (or None to estimate)
        
    Returns:
        Adjusted configuration
    """
    # Create base configuration
    config = LargeDocumentConfig()
    
    # Estimate available memory if not provided
    if available_memory is None:
        # Rough estimate (50% of system memory in a conservative approach)
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / 1024 / 1024 * 0.5
        except ImportError:
            # If psutil not available, use a conservative default
            available_memory = 1000  # Assume 1GB available
    
    # Estimate memory requirements
    memory_estimate = estimate_memory_requirements(text_length)
    
    # Adjust segment size based on document size and available memory
    if memory_estimate['total'] > available_memory:
        # Document is too large to process at once
        memory_ratio = available_memory / max(1, memory_estimate['total'])
        adjusted_segment_size = int(config.optimal_segment_size * memory_ratio)
        
        # Ensure minimum segment size
        config.optimal_segment_size = max(10000, adjusted_segment_size)
        
        # Adjust overlap proportionally
        config.segment_overlap = max(1000, int(config.segment_overlap * memory_ratio))
    
    # Adjust workers based on document size
    if text_length > config.very_large_doc_threshold:
        # For very large documents, use more workers
        config.max_workers = max(2, min(multiprocessing.cpu_count() - 1, 12))
    
    # Adjust batch size for efficiency
    if text_length > config.very_large_doc_threshold:
        config.batch_size = 10  # Smaller batches for very large documents
    
    return config