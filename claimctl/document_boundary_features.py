"""
Document boundary features module.

This module provides functions to extract and score features for document boundary
detection, supporting multi-strategy boundary detection approaches.
"""

import re
import logging
from enum import Enum
from typing import List, Dict, Set, Any, Tuple, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features for document boundary detection."""
    
    # Semantic features
    EMBEDDING_SIMILARITY = "embedding_similarity"
    TOPIC_SHIFT = "topic_shift"
    
    # Layout features
    PAGE_BREAK = "page_break"
    LARGE_WHITESPACE = "large_whitespace"
    HORIZONTAL_LINE = "horizontal_line"
    
    # Text pattern features
    DOCUMENT_HEADER = "document_header"
    DATE_HEADER = "date_header"
    EMAIL_HEADER = "email_header"
    NUMBERED_SECTION = "numbered_section"
    PAGE_NUMBER = "page_number"
    
    # Visual features
    FONT_SIZE_CHANGE = "font_size_change"
    ALIGNMENT_CHANGE = "alignment_change"


def get_boundary_features(
    text: str,
    segment_positions: List[int],
    similarities: List[float],
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Extract features for potential document boundaries.
    
    Args:
        text: Document text
        segment_positions: Start positions of segments in the text
        similarities: Similarity scores between segments
        config: Configuration dictionary
        
    Returns:
        List of feature dictionaries for potential boundaries
    """
    features = []
    
    # Initialize configuration
    if config is None:
        config = {}
    
    # Process each segment boundary
    for i in range(len(similarities)):
        # Position in text where this segment ends/next segment begins
        end_pos = segment_positions[i] + (segment_positions[i+1] - segment_positions[i]) // 2
        
        # Extract context around this position
        context_start = max(0, end_pos - 200)
        context_end = min(len(text), end_pos + 200)
        context = text[context_start:context_end]
        
        # Initialize feature set for this position
        feature_types = set()
        feature_scores = {}
        
        # Check semantic features
        similarity = similarities[i]
        similarity_score = 1.0 - similarity  # Lower similarity = higher boundary score
        
        if similarity_score > 0.3:  # Threshold for considering this a semantic boundary
            feature_types.add(FeatureType.EMBEDDING_SIMILARITY)
            feature_scores[FeatureType.EMBEDDING_SIMILARITY] = similarity_score
        
        # Check for topic shift using semantic similarity trend
        if i > 0 and i < len(similarities) - 1:
            prev_sim = similarities[i-1]
            next_sim = similarities[i+1]
            
            # If this point is a local minimum in similarity
            if similarity < prev_sim and similarity < next_sim:
                feature_types.add(FeatureType.TOPIC_SHIFT)
                topic_shift_score = min(1.0, 0.5 + (min(prev_sim, next_sim) - similarity))
                feature_scores[FeatureType.TOPIC_SHIFT] = topic_shift_score
        
        # Check layout features
        
        # Page breaks
        if '\f' in context:
            feature_types.add(FeatureType.PAGE_BREAK)
            # Score higher if page break is near center of context
            page_break_pos = context.find('\f')
            centrality = 1.0 - abs(page_break_pos - len(context)/2) / (len(context)/2)
            feature_scores[FeatureType.PAGE_BREAK] = 0.6 + 0.4 * centrality
        
        # Large whitespace
        if re.search(r'\n\s*\n\s*\n', context):
            feature_types.add(FeatureType.LARGE_WHITESPACE)
            # Count newlines to score whitespace size
            # Find the first triple newline pattern
            triple_newline_pos = context.find('\n')
            if '\n \n \n' in context:
                triple_newline_pos = context.find('\n \n \n')
            elif '\n\n\n' in context:
                triple_newline_pos = context.find('\n\n\n')

            # Count newlines up to that position
            whitespace_count = len(re.findall(r'\n', context[:triple_newline_pos+10]))
            feature_scores[FeatureType.LARGE_WHITESPACE] = min(0.8, 0.4 + 0.1 * whitespace_count)
        
        # Horizontal lines
        if re.search(r'[-=_]{10,}', context):
            feature_types.add(FeatureType.HORIZONTAL_LINE)
            feature_scores[FeatureType.HORIZONTAL_LINE] = 0.7
        
        # Check text pattern features
        
        # Document headers
        doc_header_patterns = [
            r'(?i)^(?:subject|to|from|date):', # Email headers
            r'(?i)^(?:meeting minutes|daily report|invoice|change order)', # Document type headers
            r'(?i)^\s*(?:REPORT|CONTRACT|AGREEMENT|PROPOSAL|INVOICE)', # ALL CAPS headers
        ]
        
        for pattern in doc_header_patterns:
            if re.search(pattern, context, re.MULTILINE):
                feature_types.add(FeatureType.DOCUMENT_HEADER)
                feature_scores[FeatureType.DOCUMENT_HEADER] = 0.9
                break
        
        # Date headers
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}', # MM/DD/YYYY
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}' # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, context[:100], re.IGNORECASE):  # Only in first part of context
                feature_types.add(FeatureType.DATE_HEADER)
                feature_scores[FeatureType.DATE_HEADER] = 0.7
                break
        
        # Email headers
        email_patterns = [
            r'(?i)from:.*@', # From: with email
            r'(?i)to:.*@', # To: with email
            r'(?i)subject:.*\n', # Subject: line
        ]
        
        email_header_count = 0
        for pattern in email_patterns:
            if re.search(pattern, context, re.MULTILINE):
                email_header_count += 1
        
        if email_header_count >= 2:  # At least two email header elements
            feature_types.add(FeatureType.EMAIL_HEADER)
            feature_scores[FeatureType.EMAIL_HEADER] = 0.9
        
        # Numbered sections
        if re.search(r'(?:\n|^)\s*(?:\d+\.|\([A-Za-z]\)|\([0-9]\)|[IVX]+\.)\s+[A-Z]', context):
            feature_types.add(FeatureType.NUMBERED_SECTION)
            feature_scores[FeatureType.NUMBERED_SECTION] = 0.6
        
        # Page numbers
        if re.search(r'(?:Page|PAGE)\s+\d+\s+(?:of|OF)\s+\d+', context):
            feature_types.add(FeatureType.PAGE_NUMBER)
            feature_scores[FeatureType.PAGE_NUMBER] = 0.5
        
        # Calculate overall confidence
        confidence = score_boundary_confidence(feature_types, feature_scores)
        
        # Store features for this position
        features.append({
            'position': end_pos,
            'confidence': confidence,
            'feature_types': feature_types,
            'feature_scores': feature_scores
        })
    
    return features


def score_boundary_confidence(
    feature_types: Set[FeatureType],
    feature_scores: Dict[FeatureType, float]
) -> float:
    """
    Calculate overall confidence score for a potential boundary.
    
    Args:
        feature_types: Set of detected feature types
        feature_scores: Dictionary of scores for each feature type
        
    Returns:
        Overall confidence score (0-1)
    """
    if not feature_types:
        return 0.0
    
    # Feature importance weights
    weights = {
        FeatureType.EMBEDDING_SIMILARITY: 0.6,
        FeatureType.TOPIC_SHIFT: 0.5,
        FeatureType.PAGE_BREAK: 0.4,
        FeatureType.LARGE_WHITESPACE: 0.3,
        FeatureType.HORIZONTAL_LINE: 0.4,
        FeatureType.DOCUMENT_HEADER: 0.8,
        FeatureType.DATE_HEADER: 0.7,
        FeatureType.EMAIL_HEADER: 0.9,
        FeatureType.NUMBERED_SECTION: 0.5,
        FeatureType.PAGE_NUMBER: 0.3,
        FeatureType.FONT_SIZE_CHANGE: 0.5,
        FeatureType.ALIGNMENT_CHANGE: 0.4
    }
    
    # Calculate weighted score
    total_weight = 0.0
    weighted_score = 0.0
    
    for feature in feature_types:
        weight = weights.get(feature, 0.5)
        score = feature_scores.get(feature, 0.5)
        
        weighted_score += weight * score
        total_weight += weight
    
    # Handle case with no valid features
    if total_weight == 0:
        return 0.0
    
    # Calculate final score
    final_score = weighted_score / total_weight
    
    # Bonus for multiple feature types
    feature_count_bonus = min(0.3, 0.1 * (len(feature_types) - 1))
    final_score = min(0.99, final_score + feature_count_bonus)
    
    return final_score


def extract_document_metadata(text: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from document text based on detected features.
    
    Args:
        text: Document text
        features: Feature information for this document
        
    Returns:
        Dictionary of metadata
    """
    metadata = {}
    
    # Extract document type
    feature_types = features.get('feature_types', set())
    
    if FeatureType.EMAIL_HEADER in feature_types:
        metadata['doc_type'] = 'email'
    elif FeatureType.DOCUMENT_HEADER in feature_types:
        # Extract document type from header
        header_match = re.search(r'(?i)(meeting minutes|daily report|invoice|change order|request for information|submittal)', text[:500])
        if header_match:
            metadata['doc_type'] = header_match.group(1).lower().replace(' ', '_')
        else:
            metadata['doc_type'] = 'document'
    else:
        metadata['doc_type'] = 'unknown'
    
    # Extract date if present
    date_match = re.search(r'(?:Date|DATE):\s*(\d{1,2}/\d{1,2}/\d{2,4})', text[:500])
    if date_match:
        metadata['date'] = date_match.group(1)
    else:
        # Try to find any date pattern
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})'
        date_match = re.search(date_pattern, text[:500])
        if date_match:
            metadata['date'] = date_match.group(1)
    
    # Extract subject for emails
    if metadata['doc_type'] == 'email':
        subject_match = re.search(r'(?i)Subject:\s*(.+)(?:\n|$)', text[:500])
        if subject_match:
            metadata['subject'] = subject_match.group(1).strip()
    
    # Extract project name if present
    project_match = re.search(r'(?i)Project(?:\s+Name)?:\s*(.+)(?:\n|$)', text[:1000])
    if project_match:
        metadata['project'] = project_match.group(1).strip()
    
    return metadata