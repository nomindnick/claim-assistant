"""
Multi-strategy document boundary detection.

This module implements a multi-strategy approach to document boundary
detection, combining embedding-based semantic analysis with layout and
content-based features.
"""

import re
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from claimctl.document_boundary_features import (
    get_boundary_features,
    score_boundary_confidence,
    FeatureType
)

# Setup logging
logger = logging.getLogger(__name__)


class BoundaryDetectionStrategy(Enum):
    """Strategies for document boundary detection."""
    
    SEMANTIC = "semantic"  # Embedding-based semantic similarity
    LAYOUT = "layout"      # Document layout features
    CONTENT = "content"    # Content-based patterns
    HYBRID = "hybrid"      # Combination of all strategies
    ADAPTIVE = "adaptive"  # Dynamically selects best strategy


@dataclass
class BoundaryDetectionConfig:
    """Configuration for boundary detection."""
    
    strategy: BoundaryDetectionStrategy = BoundaryDetectionStrategy.HYBRID
    segment_size: int = 100
    segment_stride: int = 50
    threshold_multiplier: float = 1.0
    min_confidence: float = 0.5
    min_boundary_distance: int = 1000
    min_document_length: int = 500
    visualize: bool = False
    visualization_path: Optional[str] = None


def create_text_segments(text: str, segment_size: int = 100, 
                         segment_stride: int = 50) -> List[Tuple[str, int]]:
    """
    Split text into overlapping segments.
    
    Args:
        text: Text to segment
        segment_size: Maximum size of each segment in characters
        segment_stride: Stride between segment starts in characters
        
    Returns:
        List of (segment_text, start_position) tuples
    """
    if not text:
        return []
        
    segments = []
    for start in range(0, len(text), segment_stride):
        end = min(start + segment_size, len(text))
        segment = text[start:end]
        segments.append((segment, start))
        
        if end == len(text):
            break
            
    return segments


def get_embeddings(segments: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Get embeddings for text segments.
    
    This function attempts to use the embedding function from the main
    ingest module. If that's not available, use a mock implementation.
    
    Args:
        segments: List of text segments
        config: Configuration dictionary
        
    Returns:
        2D numpy array of embeddings
    """
    try:
        # Try to import and use the main embedding function
        from claimctl.ingest import get_embeddings as ingest_get_embeddings
        return ingest_get_embeddings(segments)
    except ImportError:
        logger.warning("Could not import embeddings function from ingest module. Using mock embeddings.")
        
        # Mock implementation for testing
        import numpy as np
        dim = 5  # Small embedding dimension for testing
        embeddings = np.random.rand(len(segments), dim)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)
        
        return embeddings


def calculate_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarities between adjacent embeddings.
    
    Args:
        embeddings: 2D array of embedding vectors
        
    Returns:
        1D array of similarity scores
    """
    if len(embeddings) <= 1:
        return np.array([])
        
    # Calculate similarities using dot product (embeddings are normalized)
    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    
    return similarities


def find_potential_boundaries(similarities: np.ndarray,
                             threshold_multiplier: float = 1.0) -> List[int]:
    """
    Find potential document boundaries based on similarity drops.

    Args:
        similarities: 1D array of similarity scores
        threshold_multiplier: Controls sensitivity (higher = fewer boundaries)

    Returns:
        List of indices where similarity drops below threshold
    """
    if len(similarities) == 0:
        print("No similarities provided, returning empty boundaries list")
        return []

    # Calculate mean and standard deviation
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    # Set threshold based on mean and standard deviation
    threshold = mean_similarity - (threshold_multiplier * std_similarity)
    print(f"Similarity stats: mean={mean_similarity:.3f}, std={std_similarity:.3f}, threshold={threshold:.3f}")

    # Find potential boundaries
    potential_boundaries = []

    # Additional approach: Find local minima
    indices_below_threshold = [i for i, sim in enumerate(similarities) if sim < threshold]
    print(f"Found {len(indices_below_threshold)} points below threshold")
    potential_boundaries.extend(indices_below_threshold)

    # Also look for significant local minima
    window_size = 3
    if len(similarities) > window_size * 2:
        for i in range(window_size, len(similarities) - window_size):
            window = similarities[i-window_size:i+window_size+1]
            # If this point is a local minimum and is less than the mean
            if similarities[i] == min(window) and similarities[i] < mean_similarity:
                potential_boundaries.append(i)

    # Remove duplicates
    potential_boundaries = list(set(potential_boundaries))
    potential_boundaries.sort()

    # Print detailed info about detected boundaries
    if potential_boundaries:
        print("Potential boundary details:")
        for i, idx in enumerate(potential_boundaries):
            sim_val = similarities[idx]
            print(f"  #{i+1}: Position {idx}, Similarity: {sim_val:.3f}, Diff from mean: {mean_similarity - sim_val:.3f}")

    return potential_boundaries


def refine_boundaries(text: str, potential_boundaries: List[int],
                     segment_positions: List[int], similarities: List[float],
                     min_boundary_distance: int = 1000,
                     min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Refine boundary positions and calculate confidence.

    Args:
        text: Document text
        potential_boundaries: Indices of potential boundaries
        segment_positions: Start positions of segments in the text
        similarities: Similarity scores between segments
        min_boundary_distance: Minimum distance between boundaries
        min_confidence: Minimum confidence threshold for boundaries

    Returns:
        List of refined boundary dictionaries
    """
    if not potential_boundaries:
        print("No potential boundaries to refine")
        return []

    # Map segment indices to text positions
    print(f"Segment indices to refine: {potential_boundaries}")

    # Convert segment indices to positions in text
    boundary_positions = []
    for idx in potential_boundaries:
        if idx < len(segment_positions) - 1:
            segment_start = segment_positions[idx]
            segment_end = segment_positions[idx + 1]
            # Use middle of the segment as the boundary position
            boundary_position = segment_start + (segment_end - segment_start) // 2
            boundary_positions.append(boundary_position)

    if not boundary_positions:
        print("Failed to convert segment indices to boundary positions")
        return []

    print(f"Converted to text positions: {boundary_positions}")

    # Calculate confidence for each boundary position
    refined_boundaries = []
    prev_position = -min_boundary_distance  # Initialize with negative to allow first boundary

    # Sort boundary positions
    boundary_positions.sort()

    for i, position in enumerate(boundary_positions):
        # Find corresponding similarity value
        idx = potential_boundaries[i] if i < len(potential_boundaries) else -1
        similarity = similarities[idx] if idx >= 0 and idx < len(similarities) else 0.5

        # Create context around the boundary for feature extraction
        context_start = max(0, position - 200)
        context_end = min(len(text), position + 200)
        context = text[context_start:context_end]

        # Calculate confidence directly using the score_boundary function
        # 1 - similarity is higher confidence (lower similarity = bigger semantic shift)
        confidence = score_boundary_position(text, position, 1.0 - similarity)

        print(f"Boundary at position {position}, similarity={similarity:.3f}, confidence={confidence:.3f}")

        # Check confidence and distance constraints
        if confidence >= min_confidence and position - prev_position >= min_boundary_distance:
            # Find better position near the detected boundary
            refined_position = refine_boundary_position(text, position)

            # Update boundary information
            boundary = {
                'position': refined_position,
                'confidence': confidence,
                'original_index': idx,
                'similarity': similarity
            }

            refined_boundaries.append(boundary)
            prev_position = refined_position
            print(f"Added boundary at position {refined_position} with confidence {confidence:.3f}")
        else:
            if confidence < min_confidence:
                print(f"Rejected boundary at position {position}: confidence {confidence:.3f} < threshold {min_confidence}")
            else:
                print(f"Rejected boundary at position {position}: too close to previous boundary at {prev_position}")

    return refined_boundaries


def score_boundary_position(text: str, position: int, similarity_score: float) -> float:
    """
    Score a boundary position based on semantic and textual features.

    Args:
        text: Document text
        position: Boundary position in text
        similarity_score: Base score from semantic similarity (higher = better boundary)

    Returns:
        Confidence score (0-1)
    """
    # Get context around the position
    context_start = max(0, position - 200)
    context_end = min(len(text), position + 200)
    context = text[context_start:context_end]

    # Start with similarity score as base
    base_score = min(0.8, similarity_score)

    # Check for boundary indicators
    indicators = {
        # Document headers
        r'(?i)^\s*(subject|date|from|to):': 0.3,
        r'(?i)^\s*(meeting minutes|daily report|change order)': 0.3,

        # Layout indicators
        r'\f': 0.2,  # Form feed (page break)
        r'\n\s*\n\s*\n': 0.2,  # Multiple blank lines
        r'[-=_]{10,}': 0.2,  # Horizontal lines

        # Numbered sections or headers
        r'(?:\n|^)\s*\d+\.\s+[A-Z]': 0.15,
        r'(?:\n|^)\s*[IVX]+\.\s+[A-Z]': 0.15,

        # Date headers
        r'\d{1,2}/\d{1,2}/\d{2,4}': 0.1
    }

    bonus = 0.0
    for pattern, value in indicators.items():
        if re.search(pattern, context):
            bonus += value

    # Cap the bonus
    bonus = min(0.5, bonus)

    # Combine scores
    final_score = min(0.95, base_score + bonus)

    return final_score

def refine_boundary_position(text: str, position: int, window: int = 100) -> int:
    """
    Refine boundary position by finding a better break nearby.

    Args:
        text: Document text
        position: Initial boundary position
        window: Search window size

    Returns:
        Refined boundary position
    """
    # Define maximum position to avoid index errors
    max_pos = len(text) - 1
    
    # Define search window
    start = max(0, position - window // 2)
    end = min(max_pos, position + window // 2)
    
    # Search for a good break point
    best_pos = position
    best_score = 0
    
    # Patterns in descending order of preference
    patterns = [
        r"\n\s*\n",     # Double newline (paragraph break)
        r"\.\s*\n",     # End of sentence then newline
        r"\n",          # Single newline
        r"\.\s",        # End of sentence
        r";\s",         # Semicolon
        r",\s",         # Comma
        r"\s"           # Any whitespace
    ]
    
    # Search for each pattern
    search_text = text[start:end]
    for i, pattern in enumerate(patterns):
        matches = list(re.finditer(pattern, search_text))
        if matches:
            # Calculate scores based on pattern quality and distance from original position
            for match in matches:
                match_pos = start + match.start()
                distance = abs(match_pos - position)
                pattern_quality = len(patterns) - i  # Higher value for better patterns
                score = pattern_quality * (1.0 - distance / window)
                
                if score > best_score:
                    best_score = score
                    best_pos = match_pos
            
            # If we found any match with this pattern, stop searching
            if best_score > 0:
                break
    
    return best_pos


def detect_document_boundaries_multi_strategy(
    text: str,
    config: Optional[Dict[str, Any]] = None,
    strategy: BoundaryDetectionStrategy = BoundaryDetectionStrategy.HYBRID,
    segment_size: int = 100,
    segment_stride: int = 50,
    threshold_multiplier: float = 1.0,
    min_confidence: float = 0.5,
    min_boundary_distance: int = 1000,
    min_document_length: int = 500,
    pdf_path: Optional[str] = None,
    visualize: bool = False,
    visualization_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Detect document boundaries using multi-strategy approach.
    
    Args:
        text: Document text
        config: Configuration dictionary
        strategy: Boundary detection strategy
        segment_size: Size of text segments for embedding
        segment_stride: Stride between segments
        threshold_multiplier: Sensitivity control
        min_confidence: Minimum confidence for boundaries
        min_boundary_distance: Minimum character distance between boundaries
        min_document_length: Minimum length for a valid document
        pdf_path: Path to PDF file for additional visual layout analysis
        visualize: Whether to create visualization
        visualization_path: Path to save visualization
        
    Returns:
        List of boundary dictionaries
    """
    # Handle empty or very short text
    if not text or len(text) < min_document_length:
        return []
    
    # Create configuration object
    detection_config = BoundaryDetectionConfig(
        strategy=strategy,
        segment_size=segment_size,
        segment_stride=segment_stride,
        threshold_multiplier=threshold_multiplier,
        min_confidence=min_confidence,
        min_boundary_distance=min_boundary_distance,
        min_document_length=min_document_length,
        visualize=visualize,
        visualization_path=visualization_path
    )
    
    # Create text segments
    segments_with_positions = create_text_segments(
        text, segment_size=segment_size, segment_stride=segment_stride
    )
    
    if not segments_with_positions:
        return []
    
    segment_texts, segment_positions = zip(*segments_with_positions)
    
    # Get embeddings for semantic strategy
    boundaries = []
    
    if strategy in [BoundaryDetectionStrategy.SEMANTIC, BoundaryDetectionStrategy.HYBRID, 
                    BoundaryDetectionStrategy.ADAPTIVE]:
        # Use semantic strategy (embedding-based)
        try:
            print(f"Generating embeddings for {len(segment_texts)} segments...")
            embeddings = get_embeddings(segment_texts, config or {})
            if len(embeddings) > 0:
                print(f"Embeddings shape: {embeddings.shape}")
            else:
                print("Warning: Empty embeddings array returned")

            similarities = calculate_similarities(embeddings)
            print(f"Calculated {len(similarities)} similarity scores")
            if len(similarities) > 0:
                print(f"Similarity range: {similarities.min():.3f} to {similarities.max():.3f}, mean: {similarities.mean():.3f}")
        except Exception as e:
            print(f"Error in embeddings processing: {str(e)}")
            # Create random embeddings as fallback
            print("Using random embeddings as fallback")
            embeddings = np.random.rand(len(segment_texts), 384)
            similarities = calculate_similarities(embeddings)
        
        # Find potential boundaries with more debug output
        print(f"Finding potential boundaries with threshold_multiplier={threshold_multiplier}...")
        potential_boundaries = find_potential_boundaries(
            similarities, threshold_multiplier=threshold_multiplier
        )
        print(f"Found {len(potential_boundaries)} potential boundaries")
        
        print(f"Refining boundaries with min_confidence={min_confidence}, min_boundary_distance={min_boundary_distance}...")
        semantic_boundaries = refine_boundaries(
            text, potential_boundaries, segment_positions, similarities,
            min_boundary_distance=min_boundary_distance,
            min_confidence=min_confidence
        )
        print(f"Refined to {len(semantic_boundaries)} semantic boundaries")
        
        boundaries.extend(semantic_boundaries)
    
    if strategy in [BoundaryDetectionStrategy.LAYOUT, BoundaryDetectionStrategy.CONTENT, 
                   BoundaryDetectionStrategy.HYBRID, BoundaryDetectionStrategy.ADAPTIVE]:
        # Extract layout and content features even without semantic boundaries
        # Include PDF path in config if available
        combined_config = config or {}
        if pdf_path:
            combined_config = {**combined_config, 'pdf_path': pdf_path}

        layout_content_features = get_boundary_features(
            text, segment_positions,
            # If we don't have similarities, use a default value
            similarities=[0.5] * (len(segment_positions) - 1) if 'similarities' not in locals() else similarities,
            config=combined_config
        )
        
        # Filter by feature types based on strategy
        filtered_features = []
        for features in layout_content_features:
            feature_types = features.get('feature_types', set())
            
            if (strategy == BoundaryDetectionStrategy.LAYOUT and
                any(ft in feature_types for ft in [
                    FeatureType.PAGE_BREAK, FeatureType.HORIZONTAL_LINE,
                    FeatureType.LARGE_WHITESPACE, FeatureType.ALIGNMENT_CHANGE,
                    FeatureType.FONT_SIZE_CHANGE
                ])):
                filtered_features.append(features)
                
            elif (strategy == BoundaryDetectionStrategy.CONTENT and
                 any(ft in feature_types for ft in [
                     FeatureType.DOCUMENT_HEADER, FeatureType.EMAIL_HEADER,
                     FeatureType.DATE_HEADER, FeatureType.NUMBERED_SECTION,
                     FeatureType.PAGE_NUMBER, FeatureType.TOPIC_SHIFT
                 ])):
                filtered_features.append(features)
                
            elif strategy in [BoundaryDetectionStrategy.HYBRID, BoundaryDetectionStrategy.ADAPTIVE]:
                filtered_features.append(features)
                
        # Find positions of potential boundaries
        potential_indices = []
        for i, features in enumerate(layout_content_features):
            if features.get('confidence', 0.0) >= min_confidence:
                potential_indices.append(i)
                
        # Refine boundaries
        additional_boundaries = refine_boundaries(
            text, potential_indices, segment_positions,
            # If we don't have similarities, use a default value
            similarities=[0.5] * (len(segment_positions) - 1) if 'similarities' not in locals() else similarities,
            min_boundary_distance=min_boundary_distance,
            min_confidence=min_confidence
        )
        
        boundaries.extend(additional_boundaries)
    
    # Remove duplicates and sort by position
    unique_boundaries = {}
    for boundary in boundaries:
        position = boundary['position']
        
        # If this position is not already in unique_boundaries, or if this boundary
        # has higher confidence than the existing one, update it
        if (position not in unique_boundaries or
            boundary['confidence'] > unique_boundaries[position]['confidence']):
            unique_boundaries[position] = boundary
            
    # Filter out boundaries that would create documents shorter than min_document_length
    filtered_boundaries = []
    prev_position = 0
    
    for position in sorted(unique_boundaries.keys()):
        # Check if distance from previous boundary is sufficient
        if position - prev_position >= min_document_length:
            filtered_boundaries.append(unique_boundaries[position])
            prev_position = position
            
    # Check if last document is long enough (distance from last boundary to end of text)
    if filtered_boundaries and len(text) - filtered_boundaries[-1]['position'] < min_document_length:
        filtered_boundaries.pop()
    
    # Create visualization if requested
    if visualize:
        visualize_boundaries(
            text, segment_positions, 
            similarities if 'similarities' in locals() else [0.5] * (len(segment_positions) - 1),
            filtered_boundaries, visualization_path
        )
    
    return filtered_boundaries


def visualize_boundaries(text: str, segment_positions: List[int], 
                        similarities: List[float], boundaries: List[Dict[str, Any]],
                        output_path: Optional[str] = None):
    """
    Visualize document boundaries with confidence scores.
    
    Args:
        text: Document text
        segment_positions: Start positions of segments
        similarities: Similarity scores between segments
        boundaries: Detected boundary information
        output_path: Path to save visualization
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot similarities
    segment_indices = range(len(similarities))
    ax1.plot(segment_indices, similarities, 'b-', alpha=0.7)
    ax1.set_ylabel('Similarity')
    ax1.set_title('Document Boundary Detection')
    
    # Add mean line
    mean_similarity = np.mean(similarities)
    ax1.axhline(y=mean_similarity, color='g', linestyle='--', alpha=0.7, label='Mean')
    
    # Add threshold line
    std_similarity = np.std(similarities)
    threshold = mean_similarity - std_similarity
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
    
    # Mark boundaries on similarity plot
    for boundary in boundaries:
        # Find the closest segment index
        position = boundary['position']
        original_index = boundary.get('original_index', -1)
        
        if original_index >= 0 and original_index < len(similarities):
            ax1.plot(original_index, similarities[original_index], 'ro', markersize=8)
            
    # Plot boundary confidence
    positions = []
    confidences = []
    
    for boundary in boundaries:
        position = boundary['position']
        confidence = boundary['confidence']
        
        # Convert position to relative position in the text
        rel_position = position / max(1, len(text))
        positions.append(rel_position)
        confidences.append(confidence)
    
    # Create bottom plot for confidence
    if positions:
        ax2.bar(positions, confidences, width=0.01, color='r', alpha=0.7)
        
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Confidence')
    ax2.set_xlabel('Relative Position in Text')
    
    # Add legend
    ax1.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


# Make the multi-strategy function the default
detect_document_boundaries = detect_document_boundaries_multi_strategy