"""
Integrated document processing pipeline with boundary detection and chunking.

This module provides a complete pipeline for processing documents, including
boundary detection, document segmentation, and intelligent chunking.
"""

import os
import time
import logging
from typing import List, Dict, Tuple, Any, Optional, Iterator
from pathlib import Path
import json

# Import components
from claimctl.document_boundary_features import (
    get_boundary_features,
    score_boundary_confidence,
    FeatureType
)
from claimctl.multi_strategy_boundary_detection import (
    detect_document_boundaries,
    BoundaryDetectionStrategy
)
from claimctl.large_document_processor import (
    process_large_document,
    is_large_document,
    LargeDocumentConfig,
    adjust_processing_parameters
)

# Setup logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process documents with boundary detection and chunking.
    
    This class provides methods to:
    1. Detect document boundaries in large PDFs
    2. Split PDFs into logical documents
    3. Apply appropriate chunking strategy to each document
    4. Handle very large documents efficiently
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize boundary detection parameters
        self.boundary_config = {
            'strategy': self.config.get('boundary_strategy', 'hybrid'),
            'segment_size': self.config.get('segment_size', 100),
            'segment_stride': self.config.get('segment_stride', 50),
            'threshold_multiplier': self.config.get('threshold_multiplier', 1.0),
            'min_confidence': self.config.get('min_confidence', 0.5),
            'min_boundary_distance': self.config.get('min_boundary_distance', 1000),
            'min_document_length': self.config.get('min_document_length', 500),
            'visualize': self.config.get('visualize', False),
        }
        
        # Initialize large document parameters
        self.large_doc_config = LargeDocumentConfig(
            large_doc_threshold=self.config.get('large_doc_threshold', 500000),
            very_large_doc_threshold=self.config.get('very_large_doc_threshold', 5000000),
            optimal_segment_size=self.config.get('segment_size', 100000),
            segment_overlap=self.config.get('segment_overlap', 10000),
            max_workers=self.config.get('max_workers', None),
            batch_size=self.config.get('batch_size', 25),
            force_gc=self.config.get('force_gc', True),
            similarity_threshold=self.config.get('similarity_threshold', 0.8)
        )
        
        # Initialize chunking parameters
        self.chunking_config = {
            'chunk_size': self.config.get('chunk_size', 400),
            'chunk_overlap': self.config.get('chunk_overlap', 100),
            'semantic_chunking': self.config.get('semantic_chunking', True),
            'hierarchical_chunking': self.config.get('hierarchical_chunking', True),
            'adaptive_chunking': self.config.get('adaptive_chunking', True)
        }
        
        # Set boundary detection strategy
        strategy_str = self.boundary_config['strategy'].lower()
        strategy_mapping = {
            'semantic': BoundaryDetectionStrategy.SEMANTIC,
            'layout': BoundaryDetectionStrategy.LAYOUT,
            'content': BoundaryDetectionStrategy.CONTENT,
            'hybrid': BoundaryDetectionStrategy.HYBRID,
            'adaptive': BoundaryDetectionStrategy.ADAPTIVE
        }
        self.boundary_strategy = strategy_mapping.get(
            strategy_str, BoundaryDetectionStrategy.HYBRID
        )
    
    def detect_boundaries(self, text: str, visualize_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect document boundaries in text.
        
        Args:
            text: Document text
            visualize_path: Path to save visualization (optional)
            
        Returns:
            List of detected boundaries
        """
        # Check for large documents
        if is_large_document(text, self.large_doc_config):
            logger.info(f"Detecting boundaries in large document ({len(text)} chars)")
            
            # Process large document with optimized processing
            return self._detect_boundaries_large(text, visualize_path)
        else:
            # Process regular document
            return detect_document_boundaries(
                text,
                config=self.config,
                strategy=self.boundary_strategy,
                segment_size=self.boundary_config['segment_size'],
                segment_stride=self.boundary_config['segment_stride'],
                threshold_multiplier=self.boundary_config['threshold_multiplier'],
                min_confidence=self.boundary_config['min_confidence'],
                min_boundary_distance=self.boundary_config['min_boundary_distance'],
                min_document_length=self.boundary_config['min_document_length'],
                visualize=self.boundary_config['visualize'],
                visualization_path=visualize_path
            )
    
    def _detect_boundaries_large(self, text: str, visualize_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect boundaries in large documents with optimized processing.
        
        Args:
            text: Document text
            visualize_path: Path to save visualization (optional)
            
        Returns:
            List of detected boundaries
        """
        # Adjust processing parameters based on document size
        doc_config = adjust_processing_parameters(len(text))
        
        # Define process function for segments
        def process_segment(segment_text, **kwargs):
            return detect_document_boundaries(
                segment_text,
                config=self.config,
                strategy=self.boundary_strategy,
                segment_size=self.boundary_config['segment_size'],
                segment_stride=self.boundary_config['segment_stride'],
                threshold_multiplier=self.boundary_config['threshold_multiplier'],
                min_confidence=self.boundary_config['min_confidence'],
                min_boundary_distance=self.boundary_config['min_boundary_distance'] // 2,  # Adjust for segments
                min_document_length=self.boundary_config['min_document_length'] // 2,  # Adjust for segments
                visualize=False  # No visualization for segments
            )
        
        # Process with large document handler
        boundaries = process_large_document(
            text,
            process_func=process_segment,
            process_args={},
            config=doc_config
        )
        
        # Adjust boundary positions for global context
        global_boundaries = []
        visited_positions = set()
        
        for boundary in sorted(boundaries, key=lambda x: x.get('position', 0)):
            position = boundary.get('position', 0)
            
            # Skip duplicates (may happen at segment overlaps)
            if any(abs(position - pos) < self.boundary_config['min_boundary_distance'] // 2 
                  for pos in visited_positions):
                continue
                
            visited_positions.add(position)
            global_boundaries.append(boundary)
        
        # Create visualization if requested
        if self.boundary_config['visualize'] and visualize_path:
            try:
                from claimctl.multi_strategy_boundary_detection import visualize_boundaries
                
                # Create simplified visualization for large documents
                visualize_boundaries(
                    text[:min(len(text), 1000000)],  # Limit visualization to first 1M chars
                    list(range(0, len(text), 10000))[:100],  # Sample positions
                    [0.5] * 99,  # Placeholder similarities
                    global_boundaries,
                    visualize_path
                )
            except Exception as e:
                logger.error(f"Visualization error: {e}")
        
        return global_boundaries
    
    def split_into_documents(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Split text into separate logical documents.
        
        Args:
            text: Document text
            filename: Original filename
            
        Returns:
            List of document dictionaries
        """
        # Detect boundaries
        boundaries = self.detect_boundaries(text)
        
        # Extract document texts
        documents = []
        
        # Add start boundary at position 0
        boundary_positions = [0] + [b['position'] for b in boundaries]
        
        # Add last position
        boundary_positions.append(len(text))
        
        # Create document for each boundary pair
        for i in range(len(boundary_positions) - 1):
            start_pos = boundary_positions[i]
            end_pos = boundary_positions[i+1]
            
            # Extract document text
            doc_text = text[start_pos:end_pos]
            
            # Skip empty documents
            if not doc_text.strip():
                continue
                
            # Create document info
            document = {
                'text': doc_text,
                'start_position': start_pos,
                'end_position': end_pos,
                'original_file': filename,
                'document_index': i,
                'confidence': boundaries[i-1]['confidence'] if i > 0 else 1.0
            }
            
            documents.append(document)
        
        logger.info(f"Split document into {len(documents)} logical documents")
        return documents
    
    def process_document(self, file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document file with boundary detection and chunking.
        
        Args:
            file_path: Path to document file
            output_dir: Directory to save output files (optional)
            
        Returns:
            Dictionary of processing results
        """
        start_time = time.time()
        
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Get file extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Choose appropriate extraction method based on file type
        if file_ext == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_ext in ['.txt', '.md', '.html', '.htm']:
            text = self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Split into documents
        documents = self.split_into_documents(text, filename)
        
        # Prepare output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save extracted documents
            for i, doc in enumerate(documents):
                output_path = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(filename)[0]}_doc_{i+1}{file_ext}"
                )
                
                # Save document text to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(doc['text'])
                
                # Add output path to document info
                doc['output_path'] = output_path
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
            
            # Create metadata without full text to save space
            metadata = {
                'filename': filename,
                'processing_time': time.time() - start_time,
                'document_count': len(documents),
                'documents': [{
                    'start_position': doc['start_position'],
                    'end_position': doc['end_position'],
                    'document_index': doc['document_index'],
                    'confidence': doc['confidence'],
                    'length': len(doc['text']),
                    'output_path': doc.get('output_path')
                } for doc in documents]
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        # Return processing results
        return {
            'filename': filename,
            'processing_time': time.time() - start_time,
            'document_count': len(documents),
            'documents': documents
        }
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            # Try to use PyMuPDF (fitz) for PDF extraction
            import fitz
            
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            
            return text
            
        except ImportError:
            # Fall back to alternative method
            logger.warning("PyMuPDF not available, text extraction may be limited")
            
            try:
                from claimctl.ingest import extract_pdf_text
                return extract_pdf_text(pdf_path)
            except ImportError:
                raise ImportError("No PDF extraction method available. Install PyMuPDF or configure ingest module.")
    
    def _extract_text_file(self, file_path: str) -> str:
        """
        Extract text from text-based file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Extracted text
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


def process_document_file(file_path: str, output_dir: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a document file with boundary detection and chunking.
    
    Args:
        file_path: Path to document file
        output_dir: Directory to save output files (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary of processing results
    """
    processor = DocumentProcessor(config)
    return processor.process_document(file_path, output_dir)


def process_directory(dir_path: str, output_dir: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Process all documents in a directory.
    
    Args:
        dir_path: Path to directory
        output_dir: Directory to save output files (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        List of processing results
    """
    processor = DocumentProcessor(config)
    results = []
    
    # Prepare output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in directory
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        # Skip non-document files
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.pdf', '.txt', '.md', '.html', '.htm']:
            continue
        
        try:
            # Process document
            result = processor.process_document(file_path, output_dir)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents with boundary detection")
    parser.add_argument("input", help="Input file or directory path")
    parser.add_argument("--output", "-o", help="Output directory for extracted documents")
    parser.add_argument("--strategy", choices=["semantic", "layout", "content", "hybrid", "adaptive"],
                      default="hybrid", help="Boundary detection strategy")
    parser.add_argument("--confidence", type=float, default=0.5,
                      help="Minimum confidence threshold for boundaries")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of boundary detection")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Prepare configuration
    config = {
        'boundary_strategy': args.strategy,
        'min_confidence': args.confidence,
        'visualize': args.visualize
    }
    
    # Process input path
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output) if args.output else None
    
    if os.path.isdir(input_path):
        # Process directory
        results = process_directory(input_path, output_dir, config)
        print(f"Processed {len(results)} documents from {input_path}")
    else:
        # Process single file
        result = process_document_file(input_path, output_dir, config)
        print(f"Processed {result['filename']} - found {result['document_count']} logical documents")