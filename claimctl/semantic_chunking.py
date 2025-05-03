"""Semantic chunking module for improved document partitioning."""

import logging
import re
from typing import Dict, List, Optional, Tuple

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress

from .config import get_config

console = Console()
logger = logging.getLogger(__name__)

def create_semantic_chunks(
    text: str, 
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0
) -> List[str]:
    """Create semantic chunks from text.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum chunk size (default from config)
        chunk_overlap: Chunk overlap size (default from config)
        breakpoint_threshold_type: "percentile" or "gradient"
        breakpoint_threshold_amount: Threshold value for determining breakpoints
        
    Returns:
        List of text chunks
    """
    config = get_config()
    
    # Use config values if not specified
    if chunk_size is None:
        chunk_size = config.chunking.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.chunking.CHUNK_OVERLAP
    
    # Special case for short text
    if len(text) <= chunk_size:
        return [text]
    
    try:
        # Create OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=config.openai.API_KEY,
            model=config.openai.EMBED_MODEL,
        )
        
        # Create semantic chunker
        text_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
        
        # Create documents
        docs = text_splitter.create_documents([text])
        
        # Extract chunks from documents
        chunks = [doc.page_content for doc in docs]
        
        # Apply size constraints and handle overlong chunks
        processed_chunks = []
        for chunk in chunks:
            # If chunk is too large, apply fallback character splitting
            if len(chunk) > chunk_size * 2:  # Allow some flexibility
                # Simple fallback splitting at paragraph boundaries
                paragraphs = chunk.split("\n\n")
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        if current_chunk:
                            current_chunk += "\n\n"
                        current_chunk += para
                    else:
                        if current_chunk:
                            processed_chunks.append(current_chunk)
                        current_chunk = para
                
                if current_chunk:
                    processed_chunks.append(current_chunk)
            else:
                processed_chunks.append(chunk)
        
        console.log(f"Created {len(processed_chunks)} semantic chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in semantic chunking: {str(e)}")
        console.log(f"[bold red]Error in semantic chunking: {str(e)}")
        
        # Fallback to regular chunking
        console.log("[yellow]Falling back to regular chunking")
        return fallback_chunk_text(text, chunk_size, chunk_overlap)


def fallback_chunk_text(
    text: str, 
    chunk_size: int = 400, 
    chunk_overlap: int = 100
) -> List[str]:
    """Fallback chunking method using simple character counts.
    
    This is used when semantic chunking fails and matches your current approach.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    return text_splitter.split_text(text)


def create_hierarchical_chunks(text: str) -> List[str]:
    """Create hierarchical chunks respecting document structure.
    
    This is a more advanced implementation for preserving section hierarchy.
    """
    try:
        # This requires llama-index which might not be installed
        from llama_index.core.node_parser import HierarchicalNodeParser
        from llama_index.core import Document
        
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128],  # Multiple levels of chunking
            chunk_overlap=20
        )
        document = Document(text=text)
        nodes = parser.get_nodes_from_documents([document])
        
        # Extract text from nodes
        chunks = [node.text for node in nodes]
        console.log(f"Created {len(chunks)} hierarchical chunks")
        return chunks
        
    except ImportError:
        console.log("[yellow]llama-index not installed, falling back to semantic chunking")
        return create_semantic_chunks(text)
    except Exception as e:
        logger.error(f"Error in hierarchical chunking: {str(e)}")
        console.log(f"[bold red]Error in hierarchical chunking: {str(e)}")
        return create_semantic_chunks(text)


def detect_document_structure(text: str) -> Tuple[str, float]:
    """Detect document structure and recommend the best chunking method.
    
    Args:
        text: The document text to analyze
        
    Returns:
        Tuple containing (recommended_chunking_method, confidence_score)
    """
    # Simple rule-based structure detection
    structure_signals = {
        "hierarchical": 0.0,
        "semantic": 0.0,
        "regular": 0.0
    }
    
    # Check for hierarchical document structure
    section_pattern = re.compile(r'(?:^|\n)(?:Section|§|Article|Chapter)\s+\d+', re.IGNORECASE)
    numbered_sections = re.compile(r'(?:^|\n)(?:\d+\.)+\s+[A-Z]', re.MULTILINE)
    bullet_points = re.compile(r'(?:^|\n)(?:•|\*|-)\s+', re.MULTILINE)
    
    # Check for structured formatting
    if section_pattern.findall(text):
        structure_signals["hierarchical"] += 0.5
    if numbered_sections.findall(text):
        structure_signals["hierarchical"] += 0.3
    if bullet_points.findall(text):
        structure_signals["hierarchical"] += 0.2
    
    # Check for tabular data (suggests regular chunking might be better)
    table_patterns = re.compile(r'(?:\|\s*\w+\s*\|)|(?:\+[-]+\+)')
    if table_patterns.findall(text):
        structure_signals["regular"] += 0.3
    
    # Check for semantic coherence signals
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 3:
        # If paragraphs are relatively consistent in size, semantic chunking is good
        para_lengths = [len(p) for p in paragraphs if p.strip()]
        if para_lengths:
            avg_length = sum(para_lengths) / len(para_lengths)
            variance = sum((l - avg_length) ** 2 for l in para_lengths) / len(para_lengths)
            coefficient_of_variation = (variance ** 0.5) / avg_length
            
            # Low variance suggests semantic chunking is good
            if coefficient_of_variation < 0.5:
                structure_signals["semantic"] += 0.4
            elif coefficient_of_variation < 1.0:
                structure_signals["semantic"] += 0.2
    
    # Use AI to detect complex structure (if text is long enough)
    if len(text) > 1000:
        try:
            ai_chunking_method = ai_detect_document_structure(text[:2000])  # Use the first 2000 chars
            structure_signals[ai_chunking_method] += 0.5
        except Exception as e:
            logger.error(f"Error in AI structure detection: {str(e)}")
            # Give a small boost to semantic chunking as the default fallback
            structure_signals["semantic"] += 0.1
    
    # Determine best method
    best_method = max(structure_signals.items(), key=lambda x: x[1])
    method_name, confidence = best_method
    
    # Add fallback if confidence is too low
    if confidence < 0.2:
        method_name = "semantic"  # Default to semantic chunking
        confidence = 0.2
    
    return method_name, confidence


def ai_detect_document_structure(text_sample: str) -> str:
    """Use AI to detect document structure and recommend chunking method.
    
    Args:
        text_sample: Sample text from the document
        
    Returns:
        Recommended chunking method: "hierarchical", "semantic", or "regular"
    """
    config = get_config()
    
    structure_prompt = """
    Analyze this document sample and determine its structure type to help choose the best text chunking method.
    Choose the SINGLE BEST option:
    
    1. "hierarchical" - Document has clear hierarchical structure with sections, subsections, numbered lists, etc.
    2. "semantic" - Document has natural semantic boundaries but no clear hierarchical structure
    3. "regular" - Document has irregular structure or tabular data that needs simple chunking
    
    Sample text:
    {text}
    
    RESPOND WITH ONLY ONE OF THESE WORDS: hierarchical, semantic, or regular
    """
    
    try:
        client = OpenAI(api_key=config.openai.API_KEY)
        response = client.chat.completions.create(
            model=config.openai.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a document structure analyzer that helps determine the best chunking method.",
                },
                {
                    "role": "user",
                    "content": structure_prompt.format(text=text_sample[:1000]),  # Use first 1000 chars
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Normalize the response
        if "hierarchical" in result:
            return "hierarchical"
        elif "semantic" in result:
            return "semantic"
        elif "regular" in result:
            return "regular"
        else:
            # Default to semantic if response is unclear
            return "semantic"
            
    except Exception as e:
        logger.error(f"Error in AI structure detection: {str(e)}")
        return "semantic"  # Default to semantic chunking


def create_adaptive_chunks(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    progress: Optional[Progress] = None,
) -> List[str]:
    """Create chunks adaptively based on document structure detection.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum chunk size (default from config)
        chunk_overlap: Chunk overlap size (default from config)
        progress: Optional progress bar
        
    Returns:
        List of text chunks
    """
    config = get_config()
    
    # Use config values if not specified
    if chunk_size is None:
        chunk_size = config.chunking.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.chunking.CHUNK_OVERLAP
    
    # Short text gets a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # For very large texts, use memory-optimized processing
    if len(text) > 500000:  # 500K characters threshold (about 100 pages)
        console.log(f"[bold yellow]Large document detected ({len(text)} chars). Using memory-optimized processing.")
        return process_large_document(text, chunk_size, chunk_overlap, progress)
    
    # Detect document structure and get recommended chunking method
    chunking_method, confidence = detect_document_structure(text)
    
    # Log the detected method and confidence
    console.log(f"[bold green]Detected document structure: {chunking_method} (confidence: {confidence:.2f})")
    
    # Apply the recommended chunking method
    if chunking_method == "hierarchical":
        return create_hierarchical_chunks(text)
    elif chunking_method == "semantic":
        return create_semantic_chunks(text, chunk_size, chunk_overlap)
    else:  # "regular"
        return fallback_chunk_text(text, chunk_size, chunk_overlap)


def process_large_document(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    progress: Optional[Progress] = None
) -> List[str]:
    """Process a large document in streaming fashion to optimize memory usage.
    
    This function splits the document into manageable segments, processes each
    segment separately, and then combines the results, all while maintaining
    reasonable memory usage.
    
    Args:
        text: The large document text
        chunk_size: Maximum chunk size
        chunk_overlap: Chunk overlap size
        progress: Optional progress bar
        
    Returns:
        List of text chunks from the entire document
    """
    # Initial segmentation based on natural boundaries (e.g., pages, sections)
    # This helps avoid cutting in the middle of important content
    segments = []
    
    # Try to find natural page breaks first
    page_breaks = re.split(r'\n\s*\n\s*\n', text)  # Split on double newlines
    
    # If we have too few segments, try splitting on single paragraph breaks
    if len(page_breaks) < 5:
        segments = re.split(r'\n\s*\n', text)  # Split on paragraph breaks
    else:
        segments = page_breaks
    
    # If we still don't have enough segments, force split by size
    if len(segments) < 3:
        # Calculate a segment size that's large but manageable
        segment_size = min(100000, len(text) // 5)  # Max 100K or 1/5 of text
        segments = []
        
        # Split the text into segments with overlap
        for i in range(0, len(text), segment_size - chunk_overlap):
            segment = text[i:i + segment_size]
            segments.append(segment)
    
    # Log the segmentation strategy
    console.log(f"[green]Segmented document into {len(segments)} parts for memory-efficient processing")
    
    # Create a task in the progress bar if provided
    task_id = None
    if progress is not None:
        task_id = progress.add_task("Processing large document", total=len(segments))
    
    all_chunks = []
    
    # Process each segment with the appropriate chunking method
    for i, segment in enumerate(segments):
        if progress is not None:
            progress.update(task_id, description=f"Processing segment {i+1}/{len(segments)}")
        
        # Skip empty segments
        if not segment.strip():
            continue
        
        # For each segment, detect and apply the best chunking method
        try:
            # Use a simplified detection for segments to save processing time
            segment_structure, _ = detect_document_structure(segment)
            
            # Apply appropriate chunking based on structure
            if segment_structure == "hierarchical":
                segment_chunks = create_hierarchical_chunks(segment)
            elif segment_structure == "semantic":
                segment_chunks = create_semantic_chunks(segment, chunk_size, chunk_overlap)
            else:
                segment_chunks = fallback_chunk_text(segment, chunk_size, chunk_overlap)
            
            all_chunks.extend(segment_chunks)
            
            # Update progress
            if progress is not None:
                progress.update(task_id, advance=1)
                
        except Exception as e:
            logger.error(f"Error processing segment {i}: {str(e)}")
            # Fallback to simple chunking for this segment
            fallback_chunks = fallback_chunk_text(segment, chunk_size, chunk_overlap)
            all_chunks.extend(fallback_chunks)
            
            # Update progress even on error
            if progress is not None:
                progress.update(task_id, advance=1)
    
    # Post-process chunks to eliminate potential near-duplicates from segment overlaps
    processed_chunks = remove_duplicate_chunks(all_chunks)
    
    console.log(f"[green]Completed large document processing: {len(processed_chunks)} chunks created")
    return processed_chunks


def remove_duplicate_chunks(chunks: List[str], similarity_threshold: float = 0.8) -> List[str]:
    """Remove near-duplicate chunks that may result from segmentation overlaps.
    
    Args:
        chunks: List of text chunks
        similarity_threshold: Chunks with similarity above this threshold are considered duplicates
        
    Returns:
        Deduplicated list of chunks
    """
    unique_chunks = []
    
    # A simple but effective approach for detecting near-duplicates
    # For very large documents, we can optimize this further
    for i, chunk in enumerate(chunks):
        is_duplicate = False
        
        # Compare with chunks we've already accepted
        for existing_chunk in unique_chunks:
            # Quick length-based check first (potential duplicates should be similar in length)
            if abs(len(chunk) - len(existing_chunk)) / max(len(chunk), len(existing_chunk)) < 0.2:
                # Simple character-level similarity check
                similarity = calculate_text_similarity(chunk, existing_chunk)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    # Log deduplication results
    if len(unique_chunks) < len(chunks):
        console.log(f"[green]Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
    
    return unique_chunks


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity score.
    
    This uses a character-based Jaccard similarity for speed and low memory usage.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # For long texts, only compare the beginning, middle and end
    if len(text1) > 500 and len(text2) > 500:
        # Compare first 200 chars
        start_similarity = calculate_jaccard_similarity(text1[:200], text2[:200])
        # Compare middle 200 chars
        mid1_start = max(0, len(text1) // 2 - 100)
        mid2_start = max(0, len(text2) // 2 - 100)
        middle_similarity = calculate_jaccard_similarity(
            text1[mid1_start:mid1_start + 200], 
            text2[mid2_start:mid2_start + 200]
        )
        # Compare last 200 chars
        end_similarity = calculate_jaccard_similarity(
            text1[-200:] if len(text1) > 200 else text1,
            text2[-200:] if len(text2) > 200 else text2
        )
        # Combined weighted similarity
        return (start_similarity * 0.4) + (middle_similarity * 0.2) + (end_similarity * 0.4)
    
    # For shorter texts, use full Jaccard similarity
    return calculate_jaccard_similarity(text1, text2)


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    # Convert to sets of words or character trigrams for more robustness
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    # Handle empty sets
    if not set1 or not set2:
        return 0.0
    
    # Calculate Jaccard similarity: size of intersection / size of union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0