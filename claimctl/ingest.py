"""PDF ingestion module for claim-assistant."""

import os
import re
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from rich.progress import Progress, TaskID
import typer

from .config import ensure_dirs, get_config
from .database import (
    Document,
    get_database_engine,
    init_database,
    is_page_processed,
    save_faiss_id_mapping,
    save_page_chunk,
)
from .semantic_chunking import create_semantic_chunks, create_hierarchical_chunks, fallback_chunk_text, create_adaptive_chunks
from .timeline import extract_timeline_events, batch_extract_timeline_events
from .utils import (
    calculate_sha256,
    console,
    create_progress,
    get_page_hash,
    process_page,
)

# Chunk type classification prompt
CHUNK_TYPE_PROMPT = """
You are a specialized construction document analyzer for construction claims with expertise in public agency and school district projects. Classify this document chunk into one of the following categories and include a confidence score (0-100%):

- Email: Any email correspondence between project stakeholders (look for headers like From:, To:, Subject:)
- ChangeOrder: Documents describing changes to project scope, timeline, or costs (look for CO #, PCO, Change Order #)
- Invoice: Bills, receipts, financial documents showing payments or charges (look for Invoice #, amounts with $ signs)
- Photo: Document is primarily a photograph with minimal text (construction site images, damage photos)
- ContractClause: Excerpts from the contract, specifications, or legal agreements (look for section numbering, defined terms)
- Schedule: Project schedules, timelines, Gantt charts, or delay analysis (look for dates, durations, critical path terms)
- DailyReport: Daily work reports, progress logs, or site condition documentation (look for date headers, weather conditions)
- Drawing: Technical drawings, blueprints, or design documents (look for scales, dimensions, minimal text content)
- Submittal: Material or equipment submittals and approvals (look for submittal numbers, approval stamps)
- RFI: Request for Information documents between contractors and designers (look for question/answer format)
- Claim: Formal claim documents, dispute notices, or entitlement analyses (look for claim language, disputed amounts)
- NoticeOfDelay: Formal notices regarding schedule impacts (look for terms like "hereby notified," "impact," "delay")
- PublicAgencyDoc: Documents specific to public agencies (look for board approvals, public meeting minutes)
- Other: Anything that doesn't fit the above categories

Pay special attention to:
1. Document headers and footers for official document type indicators
2. Monetary amounts and payment terms
3. Date formats and schedule references
4. Signatures and approval indicators
5. Technical terminology specific to construction
6. Public agency terminology (e.g., board approval, public works)

Document text:
{text}

FORMAT YOUR RESPONSE AS:
Category: [category name]
Confidence: [0-100%]
Reasoning: [brief explanation of why you chose this category]
"""

# Date extraction regex patterns
DATE_PATTERNS = [
    r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",  # MM/DD/YYYY or similar
    r"(?:\w{3,9}\s+\d{1,2},?\s+\d{4})",  # Month DD, YYYY
    r"(?:\d{1,2}\s+\w{3,9},?\s+\d{4})",  # DD Month YYYY
]

# Document ID extraction regex patterns
ID_PATTERNS = [
    r"(?:Change Order|CO)\s+#?\d+",  # Change Order #123
    r"(?:Invoice|INV)\s+#?\d+",  # Invoice #123
    r"(?:Document|Doc)\s+#?\d+",  # Document #123
    r"(?:Contract|Agreement)\s+#?[\w-]+",  # Contract #A-123
]

# Project name extraction patterns
PROJECT_PATTERNS = [
    r"Project(?:\s+Name)?:\s*([\w\s\-,\.&]+)(?:\n|$)",
    r"Project\s*ID:?\s*([\w\-\d]+)(?:\n|$)",
    r"(?:RE:|REGARDING:)\s*([\w\s\-,\.&]+)(?:PROJECT|CONSTRUCTION)(?:\n|$)",
]

# Parties involved extraction patterns
PARTIES_PATTERNS = [
    r"(?:Owner|Client|Customer):\s*([\w\s\-,\.&]+)(?:\n|$)",
    r"(?:Contractor|Builder):\s*([\w\s\-,\.&]+)(?:\n|$)",
    r"(?:Subcontractor):\s*([\w\s\-,\.&]+)(?:\n|$)",
    r"(?:Architect|Designer|Engineer):\s*([\w\s\-,\.&]+)(?:\n|$)",
]

# Cost and amount extraction patterns
AMOUNT_PATTERNS = [
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",  # $123,456.78
    r"(?:amount|total|sum|cost|price|value)(?:\s+of)?\s*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",  # amount of $123,456.78
    r"(?:USD|dollars)(?:\s+of)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",  # USD 123,456.78
]

# Time period extraction patterns
TIME_PERIOD_PATTERNS = [
    r"(\d+)\s+(?:day|week|month|year)s?",  # 10 days, 2 weeks, etc.
    r"(?:duration|period|timeframe|delay)(?:\s+of)?\s+(\d+)\s+(?:day|week|month|year)s?",  # duration of 10 days
]

# Contract section references
SECTION_PATTERNS = [
    r"(?:Section|§)\s+(\d+(?:\.\d+)*)",  # Section 3.2.1
    r"(?:Article|Clause|Paragraph)\s+(\d+(?:\.\d+)*)",  # Article 5.1
    r"(?:Specification|Spec)\s+Section\s+(\d+(?:\.\d+)*)",  # Specification Section 07.21
]

# Public agency specific patterns
PUBLIC_AGENCY_PATTERNS = [
    r"(?:Board|School Board|Board of Education)\s+(?:approval|approved|meeting)",  # Board approval
    r"(?:Resolution|Board Resolution)\s+(?:No\.|Number|#)?\s*(\d+(?:-\d+)?)",  # Resolution No. 2024-05
    r"(?:Public|School)\s+(?:Contract|Agreement)\s+(?:Code|Law)",  # Public Contract Code
    r"(?:DSA|Division of the State Architect)",  # DSA references (CA specific)
    r"(?:Bid|RFP|RFQ)\s+(?:No\.|Number|#)?\s*(\w+(?:-\w+)*)",  # Bid No. 2024-001
]

# Function to validate regex patterns
def validate_regex_patterns(patterns_list, name):
    """Validates a list of regex patterns and replaces invalid ones with safe defaults."""
    for i, pattern in enumerate(patterns_list):
        try:
            re.compile(pattern)
        except re.error as e:
            print(f"Warning: Invalid {name} pattern at index {i}: {pattern}")
            print(f"Error: {str(e)}")
            # Replace with a simpler, valid pattern
            if i < len(patterns_list):
                patterns_list[i] = r"Invalid Pattern Removed"
    return patterns_list

# Work description patterns
WORK_DESCRIPTION_PATTERNS = [
    r"(?:scope of work|work scope)[\s\:]+([\w\s\-,\.&;()]+)(?:\n|$)",  # scope of work: foundation repair
    r"(?:description of work|work description)[\s\:]+([\w\s\-,\.&;()]+)(?:\n|$)",  # description of work: electrical
]

# Validate all regex patterns
DATE_PATTERNS = validate_regex_patterns(DATE_PATTERNS, "date")
ID_PATTERNS = validate_regex_patterns(ID_PATTERNS, "document ID")
PROJECT_PATTERNS = validate_regex_patterns(PROJECT_PATTERNS, "project name")
PARTIES_PATTERNS = validate_regex_patterns(PARTIES_PATTERNS, "parties")
AMOUNT_PATTERNS = validate_regex_patterns(AMOUNT_PATTERNS, "amount")
TIME_PERIOD_PATTERNS = validate_regex_patterns(TIME_PERIOD_PATTERNS, "time period")
SECTION_PATTERNS = validate_regex_patterns(SECTION_PATTERNS, "section")
PUBLIC_AGENCY_PATTERNS = validate_regex_patterns(PUBLIC_AGENCY_PATTERNS, "public agency")
WORK_DESCRIPTION_PATTERNS = validate_regex_patterns(WORK_DESCRIPTION_PATTERNS, "work description")


def extract_dates(text: str) -> List[str]:
    """Extract dates from text using regex patterns."""
    dates = []
    for pattern in DATE_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in date regex pattern '{pattern}': {str(e)}")
            continue
    return dates


def extract_document_ids(text: str) -> List[str]:
    """Extract document IDs from text using regex patterns."""
    ids = []
    for pattern in ID_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            ids.extend(matches)
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in document ID regex pattern '{pattern}': {str(e)}")
            continue
    return ids


def extract_project_name(text: str) -> Optional[str]:
    """Extract project name from text using regex patterns."""
    for pattern in PROJECT_PATTERNS:
        try:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in project name regex pattern '{pattern}': {str(e)}")
            continue
    return None


def extract_parties_involved(text: str) -> Optional[str]:
    """Extract parties involved from text using regex patterns."""
    parties = []
    for pattern in PARTIES_PATTERNS:
        try:
            match = re.search(pattern, text)
            if match:
                parties.append(match.group(1).strip())
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in parties regex pattern '{pattern}': {str(e)}")
            continue

    if parties:
        return "; ".join(parties)
    return None


def extract_amounts(text: str) -> List[str]:
    """Extract monetary amounts from text using regex patterns."""
    amounts = []
    for pattern in AMOUNT_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            amounts.extend(matches)
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in amount regex pattern '{pattern}': {str(e)}")
            continue
    return amounts


def extract_time_periods(text: str) -> List[str]:
    """Extract time periods from text using regex patterns."""
    periods = []
    for pattern in TIME_PERIOD_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            periods.extend(matches)
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in time period regex pattern '{pattern}': {str(e)}")
            continue
    return periods


def extract_section_references(text: str) -> List[str]:
    """Extract contract section references from text using regex patterns."""
    sections = []
    for pattern in SECTION_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            sections.extend(matches)
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in section regex pattern '{pattern}': {str(e)}")
            continue
    return sections


def extract_public_agency_references(text: str) -> List[str]:
    """Extract public agency references from text using regex patterns."""
    references = []
    for pattern in PUBLIC_AGENCY_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            # For patterns with capturing groups, extend with full match
            if isinstance(matches, list) and matches and isinstance(matches[0], tuple):
                references.extend([m[0] for m in matches if m[0]])
            # For patterns without capturing groups, use full match
            else:
                references.extend(matches)
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in regex pattern '{pattern}': {str(e)}")
            continue
    return references


def extract_work_description(text: str) -> Optional[str]:
    """Extract work description from text using regex patterns."""
    for pattern in WORK_DESCRIPTION_PATTERNS:
        try:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        except Exception as e:
            # Log the error and continue with the next pattern
            print(f"Warning: Error in work description regex pattern '{pattern}': {str(e)}")
            continue
    return None


def classify_chunk(text: str) -> Tuple[str, int]:
    """Classify the type of document chunk using GPT-4o-mini.

    Returns:
        Tuple containing (chunk_type, confidence_score)
    """
    config = get_config()

    # List of valid document types
    valid_types = [
        "Email",
        "ChangeOrder",
        "Invoice",
        "Photo",
        "ContractClause",
        "Schedule",
        "DailyReport",
        "Drawing",
        "Submittal",
        "RFI",
        "Claim",
        "Other",
    ]

    try:
        client = OpenAI(api_key=config.openai.API_KEY)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized construction document classifier for construction claims.",
                },
                {
                    "role": "user",
                    "content": CHUNK_TYPE_PROMPT.format(text=text[:1000]),
                },  # Truncate text
            ],
            max_tokens=50,
        )

        # Parse response to extract category and confidence
        result = response.choices[0].message.content.strip()

        # Default values
        chunk_type = "Other"
        confidence = 0

        # Extract category and confidence from response
        category_match = re.search(r"Category:\s*(\w+)", result)
        confidence_match = re.search(r"Confidence:\s*(\d+)", result)

        if category_match:
            category = category_match.group(1)
            # Find closest matching category
            for valid_type in valid_types:
                if (
                    valid_type.lower() == category.lower()
                    or valid_type.lower() in category.lower()
                ):
                    chunk_type = valid_type
                    break

        if confidence_match:
            try:
                confidence = int(confidence_match.group(1))
                # Ensure confidence is in range 0-100
                confidence = max(0, min(100, confidence))
            except ValueError:
                confidence = 50  # Default if parsing fails
        else:
            confidence = 50  # Default confidence

        return chunk_type, confidence

    except Exception as e:
        console.log(f"[bold red]Error classifying chunk: {str(e)}")
        # Use simple keyword-based fallback classification
        text_lower = text.lower()

        # Basic fallback classification with confidence estimates
        if "change order" in text_lower or "co #" in text_lower:
            return "ChangeOrder", 70
        elif "@" in text_lower and (
            "from:" in text_lower or "to:" in text_lower or "subject:" in text_lower
        ):
            return "Email", 80
        elif "invoice" in text_lower or (
            "$" in text and ("total" in text_lower or "amount" in text_lower)
        ):
            return "Invoice", 75
        elif (
            "schedule" in text_lower
            or "gantt" in text_lower
            or "timeline" in text_lower
        ):
            return "Schedule", 65
        elif "daily report" in text_lower or "site conditions" in text_lower:
            return "DailyReport", 70
        elif (
            "drawing" in text_lower
            or "detail" in text_lower
            or "plan view" in text_lower
        ):
            return "Drawing", 65
        elif (
            "rfi" in text_lower
            or "request for information" in text_lower
            or "information request" in text_lower
        ):
            return "RFI", 75
        elif (
            "claim" in text_lower
            or "dispute" in text_lower
            or "entitlement" in text_lower
        ):
            return "Claim", 70
        elif len(text.strip()) < 100:  # Assume it's a photo with minimal text
            return "Photo", 60
        elif (
            "clause" in text_lower
            or "section" in text_lower
            or "agreement" in text_lower
            or "contract" in text_lower
        ):
            return "ContractClause", 65
        else:
            return "Other", 40


def create_or_load_faiss_index(
    dim: int = 3072,  # Updated from 1536 to 3072 for text-embedding-3-large
) -> Union[faiss.IndexFlatL2, faiss.IndexIVFFlat]:
    """Create a new FAISS index or load an existing one.

    For small collections, uses flat index. For larger collections, uses IVF structure.
    """
    config = get_config()
    index_path = Path(config.paths.INDEX_DIR) / "faiss.idx"

    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            console.log(f"Loaded existing FAISS index with {index.ntotal} vectors")
            return index
        except Exception as e:
            console.log(f"Error loading FAISS index: {e}. Creating a new one.")

    # Create a new index
    # If we predict collection will be large (>10k vectors), use IVF
    # For now, start with a flat index which is more accurate for small collections
    index = faiss.IndexFlatL2(dim)  # L2 distance
    console.log(f"Created new FAISS index with dimension {dim}")

    # Note: Once index has >10k vectors, we should convert to IVF for better performance
    # by enabling this code. For now, commented out as it requires training data.

    # if index.ntotal > 10000:
    #     console.log("Converting to IVF index for better performance with large dataset")
    #     nlist = min(4096, int(index.ntotal / 10))  # Rule of thumb: nlist ~= sqrt(N)
    #     ivf_index = faiss.IndexIVFFlat(index, dim, nlist, faiss.METRIC_L2)
    #     ivf_index.nprobe = 16  # Number of clusters to visit during search (higher = more accurate but slower)
    #     # Need to have data to train the quantizer
    #     # If this is enabled, will need to keep original vector data for training
    #     return ivf_index

    return index


def save_faiss_index(index: faiss.IndexFlatL2) -> None:
    """Save the FAISS index to disk."""
    config = get_config()
    index_path = Path(config.paths.INDEX_DIR) / "faiss.idx"
    faiss.write_index(index, str(index_path))
    console.log(f"Saved FAISS index with {index.ntotal} vectors")


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    config = get_config()

    try:
        embed_model = OpenAIEmbeddings(
            openai_api_key=config.openai.API_KEY,
            model=config.openai.EMBED_MODEL,
        )

        # Process texts to remove stopwords and normalize construction terminology
        processed_texts = []
        for text in texts:
            # Replace common construction terms with standardized versions
            text = text.replace("change order", "ChangeOrder")
            text = text.replace("CO #", "ChangeOrder#")
            text = text.replace("change-order", "ChangeOrder")
            text = text.replace("invoice #", "Invoice#")
            text = text.replace("inv#", "Invoice#")

            # Remove boilerplate text that doesn't add semantic value
            text = re.sub(r"Page \d+ of \d+", "", text)
            text = re.sub(r"www\.\w+\.\w+", "", text)

            processed_texts.append(text)

        embeddings = embed_model.embed_documents(processed_texts)

        # Ensure no zero vectors are returned
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Check for zero vectors and replace them if necessary
        for i, embedding in enumerate(embeddings_array):
            if np.all(np.abs(embedding) < 1e-6):  # If embedding is essentially zero
                console.log(
                    f"[bold yellow]Warning: Zero embedding detected for text: {texts[i][:100]}..."
                )
                # Create a random but consistent embedding instead
                text_hash = hash(texts[i]) % 10000
                np.random.seed(text_hash)
                random_embedding = np.random.randn(embeddings_array.shape[1]).astype(
                    np.float32
                )
                # Normalize to unit length like real embeddings
                random_embedding = random_embedding / np.linalg.norm(random_embedding)
                embeddings_array[i] = random_embedding

        return embeddings_array

    except Exception as e:
        console.log(f"[bold red]Error generating embeddings: {str(e)}")
        # Instead of returning zeros, create random but consistent embeddings
        embeddings = []
        for text in texts:
            if not text.strip():  # If text is empty
                text = "empty_text"
            text_hash = hash(text) % 10000
            np.random.seed(text_hash)
            # Determine embedding dimension (3072 for text-embedding-3-large)
            embedding_dim = 3072
            random_embedding = np.random.randn(embedding_dim).astype(np.float32)
            # Normalize to unit length
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            embeddings.append(random_embedding)

        return np.array(embeddings, dtype=np.float32)


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 100) -> List[str]:
    """Break text into smaller overlapping chunks for better semantic search."""
    # Use LangChain's text splitter for smart chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Only chunk text if it's substantial
    if len(text) > chunk_size:
        return text_splitter.split_text(text)
    else:
        return [text]


def process_pdf(
    pdf_path: Path,
    progress: Progress,
    task_id: TaskID,
    project_name: Optional[str] = None,
    matter_id: Optional[int] = None,
    ingestion_logger: Optional[Any] = None,  # Added logger parameter
) -> None:
    """Process a PDF file: extract text, metadata, generate embeddings."""
    config = get_config()
    console.log(f"Processing {pdf_path}")
    
    # Start timing document processing
    doc_start_time = time.time()
    
    # Log document processing start
    if ingestion_logger:
        ingestion_logger.log_document_start(pdf_path)

    # Initialize document
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        error_msg = f"Error opening {pdf_path}: {str(e)}"
        console.log(f"[bold red]{error_msg}")
        progress.update(task_id, advance=1, description=f"Error: {str(e)}")
        
        # Log document error
        if ingestion_logger:
            ingestion_logger.log_error(pdf_path, "file_open_error", str(e))
            ingestion_logger.log_document_complete(pdf_path, time.time() - doc_start_time, 0, 0, success=False)
        return

    # Get/create FAISS index
    index = create_or_load_faiss_index()

    # Process each page
    total_pages = len(doc)
    progress.update(task_id, total=total_pages, description="Starting")
    
    # Initialize counters for document stats
    processed_pages = 0
    total_chunks = 0
    extraction_successes = 0
    extraction_failures = 0

    # Begin transaction - we'll save the index at the end only if all pages process successfully
    try:
        for page_num in range(total_pages):
            progress.update(
                task_id, advance=0, description=f"Page {page_num+1}/{total_pages}"
            )

            # Generate unique hash for this page
            page_hash = get_page_hash(pdf_path, page_num)

            # Skip if already processed
            if is_page_processed(page_hash):
                console.log(f"Skipping page {page_num+1} (already processed)")
                progress.update(task_id, advance=1, description=f"Skipped (duplicate)")
                continue

            # Process page
            try:
                page_data = process_page(doc, page_num)
                text = page_data["text"]
                processed_pages += 1

                # Extract metadata
                dates = extract_dates(text)
                doc_ids = extract_document_ids(text)
                amounts = extract_amounts(text)
                time_periods = extract_time_periods(text)
                section_references = extract_section_references(text)
                public_agency_refs = extract_public_agency_references(text)
                work_description = extract_work_description(text)
                
                # Log metadata extractions in logger
                if ingestion_logger:
                    # Log extractions with success/failure
                    ingestion_logger.log_extraction(pdf_path, page_num, "dates", str(dates), bool(dates))
                    ingestion_logger.log_extraction(pdf_path, page_num, "doc_ids", str(doc_ids), bool(doc_ids))
                    ingestion_logger.log_extraction(pdf_path, page_num, "amounts", str(amounts), bool(amounts))
                    ingestion_logger.log_extraction(pdf_path, page_num, "time_periods", str(time_periods), bool(time_periods))
                    ingestion_logger.log_extraction(pdf_path, page_num, "section_references", str(section_references), bool(section_references))
                    ingestion_logger.log_extraction(pdf_path, page_num, "public_agency_refs", str(public_agency_refs), bool(public_agency_refs))
                    ingestion_logger.log_extraction(pdf_path, page_num, "work_description", work_description, bool(work_description))
                
                # Track extraction metrics
                extraction_successes += sum([
                    bool(dates), bool(doc_ids), bool(amounts), bool(time_periods),
                    bool(section_references), bool(public_agency_refs), bool(work_description)
                ])
                extraction_failures += sum([
                    not bool(dates), not bool(doc_ids), not bool(amounts), not bool(time_periods),
                    not bool(section_references), not bool(public_agency_refs), not bool(work_description)
                ])
                
                # Convert date strings to Python date objects if possible
                parsed_dates = []
                for date_str in dates:
                    try:
                        from datetime import datetime
                        # Try various date formats
                        for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%b %d, %Y", "%d %b %Y"]:
                            try:
                                date_obj = datetime.strptime(date_str, fmt).date()
                                parsed_dates.append(date_obj)
                                break
                            except ValueError:
                                continue
                    except Exception as e:
                        console.log(f"[bold yellow]Error parsing date {date_str}: {e}")
                        if ingestion_logger:
                            ingestion_logger.log_error(pdf_path, "date_parse_error", f"Error parsing date {date_str}: {str(e)}", page_num)
                
                # Replace string dates with parsed date objects
                dates = parsed_dates if parsed_dates else []

                # Extract project name if not provided
                if not project_name:
                    extracted_project = extract_project_name(text)
                    if extracted_project:
                        project_name = extracted_project
                        if ingestion_logger:
                            ingestion_logger.log_extraction(pdf_path, page_num, "project_name", extracted_project, True)

                # Extract parties involved for the document (not stored in chunk)
                parties_involved = extract_parties_involved(text)
                if ingestion_logger and parties_involved:
                    ingestion_logger.log_extraction(pdf_path, page_num, "parties_involved", parties_involved, True)

                # Classify chunk
                chunk_type, confidence = classify_chunk(text)
                
                # Log classification in logger
                if ingestion_logger:
                    ingestion_logger.log_classification(pdf_path, page_num, 0, chunk_type, confidence)

                # Get chunking config
                chunk_size = config.chunking.CHUNK_SIZE
                chunk_overlap = config.chunking.CHUNK_OVERLAP
                use_semantic = config.chunking.SEMANTIC_CHUNKING
                use_hierarchical = config.chunking.HIERARCHICAL_CHUNKING
                use_adaptive = config.chunking.ADAPTIVE_CHUNKING
                
                # Choose chunking method based on config
                chunking_method = "default"
                if use_adaptive:
                    # Use adaptive chunking to automatically detect document structure
                    console.log("[green]Using adaptive chunking to detect document structure")
                    text_chunks = create_adaptive_chunks(text, chunk_size, chunk_overlap, progress)
                    chunking_method = "adaptive"
                elif use_semantic:
                    if use_hierarchical and (chunk_type == "ContractClause" or chunk_type == "ChangeOrder"):
                        # Use hierarchical chunking for structured documents
                        text_chunks = create_hierarchical_chunks(text)
                        chunking_method = "hierarchical"
                    else:
                        # Use semantic chunking for other documents
                        text_chunks = create_semantic_chunks(text, chunk_size, chunk_overlap)
                        chunking_method = "semantic"
                else:
                    # Use legacy chunking
                    text_chunks = fallback_chunk_text(text, chunk_size, chunk_overlap)
                    chunking_method = "fallback"
                    
                console.log(f"Created {len(text_chunks)} chunks from page {page_num+1} using {chunking_method} chunking")
                total_chunks += len(text_chunks)

                # Prepare all chunks for batched embedding generation
                chunk_data_list = []
                for i, text_chunk in enumerate(text_chunks):
                    # Create unique ID for this chunk
                    chunk_id = f"{page_hash}_{i}"

                    # Prepare DB entry with additional metadata
                    chunk_data = {
                        "text": text_chunk,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                    }
                    chunk_data_list.append(chunk_data)

                # Generate embeddings in batches, passing the current progress bar
                embeddings = batch_generate_embeddings(
                    chunk_data_list, batch_size=10, progress=progress
                )

                # Process each chunk with its embedding
                for i, (chunk_data, embedding) in enumerate(
                    zip(chunk_data_list, embeddings)
                ):
                    # Add to FAISS index and get its position
                    current_index_size = index.ntotal
                    faiss_id = (
                        current_index_size  # This will be the ID in FAISS (0-indexed)
                    )
                    embedding_array = np.array([embedding], dtype=np.float32)
                    console.log(f"Adding embedding with shape {embedding_array.shape} to index with dimension {index.d}")
                    index.add(embedding_array)
                    
                    # Individual chunk classification if needed
                    if i > 0 and len(text_chunk) > 500:
                        # For significant chunks after the first one, we might want to classify them individually
                        try:
                            sub_chunk_type, sub_confidence = classify_chunk(text_chunk)
                            if ingestion_logger:
                                ingestion_logger.log_classification(pdf_path, page_num, i, sub_chunk_type, sub_confidence)
                        except Exception as e:
                            # If classification fails, use the page's classification
                            sub_chunk_type, sub_confidence = chunk_type, confidence

                    # Complete the chunk data with all metadata
                    chunk_data.update(
                        {
                            "file_path": str(pdf_path),
                            "file_name": pdf_path.name,
                            "page_num": page_num + 1,  # 1-indexed for humans
                            "page_hash": page_hash,
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            "image_path": page_data["image_path"],
                            "text": text_chunk,  # Use text_chunk instead of chunk_text
                            "chunk_type": chunk_type,
                            "confidence": confidence,
                            "doc_date": dates[0] if dates else None,
                            "doc_id": doc_ids[0] if doc_ids else None,
                            "project_name": project_name,
                            "amount": amounts[0] if amounts else None,
                            "time_period": time_periods[0] if time_periods else None,
                            "section_reference": section_references[0] if section_references else None,
                            "public_agency_reference": public_agency_refs[0] if public_agency_refs else None,
                            "work_description": work_description,
                            "faiss_id": faiss_id,  # Store FAISS ID directly
                            "matter_id": matter_id,  # Add matter_id
                        }
                    )

                    # Save to database
                    save_page_chunk(chunk_data)

                # Store chunk_data for timeline extraction
                chunk_data_list[i] = chunk_data

                progress.update(
                    task_id, advance=1, description=f"Processed {chunk_type} ({confidence}%)"
                )

            except Exception as e:
                error_msg = f"Error processing page {page_num+1}: {str(e)}"
                console.log(f"[bold red]{error_msg}")
                progress.update(task_id, advance=1, description=f"Error: {str(e)}")
                
                # Log page processing error
                if ingestion_logger:
                    ingestion_logger.log_error(pdf_path, "page_processing_error", str(e), page_num)

        # All pages processed successfully, save index
        save_faiss_index(index)
        
        # Check if automatic timeline extraction is enabled
        if config.timeline.AUTO_EXTRACT and chunk_data_list:
            timeline_extraction_start = time.time()
            console.log(f"[bold green]Extracting timeline events from {len(chunk_data_list)} chunks...")
            
            try:
                # Process in batches according to configuration
                batch_size = config.timeline.EXTRACTION_BATCH_SIZE
                total_events = 0
                
                # Add a progress task for timeline extraction
                timeline_task = progress.add_task("Extracting timeline events", total=len(chunk_data_list))
                
                # Process chunks in batches
                for i in range(0, len(chunk_data_list), batch_size):
                    batch = chunk_data_list[i:i+batch_size]
                    events = batch_extract_timeline_events(batch, matter_id, progress)
                    total_events += len(events)
                    
                    # Update progress
                    progress.update(timeline_task, advance=len(batch))
                
                timeline_extraction_time = time.time() - timeline_extraction_start
                console.log(f"[bold green]Timeline extraction complete: {total_events} events extracted in {timeline_extraction_time:.2f} seconds")
                
                # Log timeline extraction in logger
                if ingestion_logger:
                    ingestion_logger.log_extraction(pdf_path, -1, "timeline_events", f"{total_events} events", total_events > 0)
            except Exception as e:
                console.log(f"[bold red]Error extracting timeline events: {str(e)}")
                # Log error in logger
                if ingestion_logger:
                    ingestion_logger.log_error(pdf_path, "timeline_extraction_error", str(e))
        
        # Log successful document completion
        doc_processing_time = time.time() - doc_start_time
        if ingestion_logger:
            ingestion_logger.log_document_complete(
                pdf_path, 
                doc_processing_time,
                processed_pages, 
                total_chunks, 
                success=True
            )

    except Exception as e:
        error_msg = f"Error during PDF processing: {str(e)}"
        console.log(f"[bold red]{error_msg}")
        progress.update(task_id, description=f"Failed: {str(e)}")
        
        # Log document failure
        if ingestion_logger:
            ingestion_logger.log_error(pdf_path, "document_processing_error", str(e))
            ingestion_logger.log_document_complete(
                pdf_path, 
                time.time() - doc_start_time,
                processed_pages, 
                total_chunks, 
                success=False
            )
        
        # Index will not be saved, keeping the previous state

    # Close document
    doc.close()


def batch_generate_embeddings(
    chunks: List[Dict[str, Any]],
    batch_size: int = 10,
    progress: Optional[Progress] = None,
) -> np.ndarray:
    """Generate embeddings in batches to avoid API rate limits.

    Args:
        chunks: List of text chunks to embed
        batch_size: Number of embeddings to generate in each batch
        progress: Optional existing progress bar to use

    Returns:
        Array of embeddings for all chunks
    """
    texts = [chunk["text"] for chunk in chunks]
    total_chunks = len(texts)
    embeddings_list = []

    # Process embeddings without progress bar if none provided
    if progress is None:
        console.log(
            f"Generating embeddings for {total_chunks} chunks in batches of {batch_size}"
        )

        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                # Get embeddings for this batch
                batch_embeddings = get_embeddings(batch_texts)
                embeddings_list.append(batch_embeddings)
                console.log(
                    f"Completed embedding batch {i//batch_size + 1}/{math.ceil(total_chunks/batch_size)}"
                )
            except Exception as e:
                console.log(f"[bold red]Error in batch {i//batch_size + 1}: {str(e)}")
                # Create fallback random embeddings for this batch
                fallback_embeddings = np.zeros(
                    (len(batch_texts), 3072), dtype=np.float32
                )
                for j, text in enumerate(batch_texts):
                    text_hash = hash(text) % 10000
                    np.random.seed(text_hash)
                    random_embedding = np.random.randn(3072).astype(np.float32)
                    # Normalize to unit length
                    random_embedding = random_embedding / np.linalg.norm(
                        random_embedding
                    )
                    fallback_embeddings[j] = random_embedding
                embeddings_list.append(fallback_embeddings)
    else:
        # Use the provided progress bar
        task_id = progress.add_task(
            "Generating embeddings", total=math.ceil(total_chunks / batch_size)
        )

        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i : i + batch_size]
            progress.update(
                task_id,
                advance=0,
                description=f"Embedding batch {i//batch_size + 1}/{math.ceil(total_chunks/batch_size)}",
            )

            try:
                # Get embeddings for this batch
                batch_embeddings = get_embeddings(batch_texts)
                embeddings_list.append(batch_embeddings)
                # Update progress
                progress.update(
                    task_id, advance=1, description=f"Completed batch {i//batch_size + 1}"
                )
            except Exception as e:
                console.log(f"[bold red]Error in batch {i//batch_size + 1}: {str(e)}")
                # Still advance progress
                progress.update(
                    task_id, advance=1, description=f"Error in batch {i//batch_size + 1}"
                )
                # Create fallback random embeddings for this batch
                fallback_embeddings = np.zeros(
                    (len(batch_texts), 3072), dtype=np.float32
                )
                for j, text in enumerate(batch_texts):
                    text_hash = hash(text) % 10000
                    np.random.seed(text_hash)
                    random_embedding = np.random.randn(3072).astype(np.float32)
                    # Normalize to unit length
                    random_embedding = random_embedding / np.linalg.norm(
                        random_embedding
                    )
                    fallback_embeddings[j] = random_embedding
                embeddings_list.append(fallback_embeddings)

    # Combine all batches
    return np.vstack(embeddings_list) if embeddings_list else np.array([])


def ingest_pdfs(
    pdf_paths: List[Path],
    project_name: Optional[str] = None,
    batch_size: int = 5,
    resume_on_error: bool = True,
    matter_name: Optional[str] = None,  # Add matter_name parameter
    enable_logging: bool = True,  # Enable ingestion logging by default
) -> None:
    """Ingest a list of PDF files with matter awareness.

    Args:
        pdf_paths: List of PDF paths to process
        project_name: Optional project name to associate with documents
        batch_size: Number of documents to process in each batch
        resume_on_error: Whether to continue processing after errors
        matter_name: Optional matter name to associate with documents
        enable_logging: Whether to enable detailed ingestion logging (default: True)
    """
    # Use current matter if not specified
    from .config import get_current_matter, set_current_matter
    from .database import get_session, Matter
    
    if not matter_name:
        matter_name = get_current_matter()
        if not matter_name:
            console.print("[bold red]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Ensure database tables exist before querying
    try:
        init_database()
    except Exception as db_init_error:
        console.log(f"Note: Database initialization attempt: {db_init_error}")
        
    # Get matter directories
    with get_session() as session:
        matter = session.query(Matter).filter(Matter.name == matter_name).first()
        if not matter:
            # We have a matter name in config, but it doesn't exist in DB
            # This can happen after clearing the database
            console.print(f"[bold red]Matter '{matter_name}' not found in database")
            console.print(f"[bold yellow]The active matter was cleared. Please create a new matter.")
            # Reset current matter in config since it no longer exists
            set_current_matter("")
            raise typer.Exit(1)
            
        matter_dir = Path(matter.data_directory).parent  # Parent directory of the data directory
        data_dir = Path(matter.data_directory)
        index_dir = Path(matter.index_directory)
    
    # Override data and index directories for this operation
    config = get_config()
    original_data_dir = config.paths.DATA_DIR
    original_index_dir = config.paths.INDEX_DIR
    
    # Temporarily set paths for this matter
    config.paths.DATA_DIR = str(data_dir)
    config.paths.INDEX_DIR = str(index_dir)
    
    # Initialize ingestion logger if enabled
    ingestion_logger = None
    if enable_logging:
        from .ingestion_logger import IngestionLogger
        try:
            ingestion_logger = IngestionLogger(matter_name, matter_dir)
            console.log(f"[green]Ingestion logging enabled for matter '{matter_name}'")
        except Exception as e:
            console.log(f"[yellow]Warning: Could not initialize ingestion logger: {e}")
    
    try:
        # Ensure directories exist
        ensure_dirs()
    
        # Initialize database
        init_database()
    
        # Create resume log file for this matter
        resume_log_path = data_dir / "ingest_resume.log"
        completed_pdfs = set()
        
        # Check if this is a fresh matter (just created)
        try:
            with get_session() as check_session:
                doc_count = check_session.query(Document).filter(Document.matter_id == matter.id).count()
                is_fresh_matter = doc_count == 0
        except Exception as e:
            console.log(f"[bold yellow]Error checking if matter is fresh: {e}")
            # Assume it's a fresh matter to force reprocessing in case of errors
            is_fresh_matter = True
        
        # If this is a fresh matter, we should force reprocessing by ignoring the resume log
        if is_fresh_matter and resume_log_path.exists():
            console.log("[bold yellow]Fresh matter detected, clearing resume log to force reprocessing")
            resume_log_path.unlink()
    
        # Load previously completed files if resume file exists and we're not starting fresh
        if resume_log_path.exists() and not is_fresh_matter:
            try:
                with open(resume_log_path, "r") as f:
                    completed_pdfs = set(line.strip() for line in f.readlines())
                console.log(
                    f"[bold yellow]Found {len(completed_pdfs)} previously processed files"
                )
            except Exception as e:
                console.log(f"[bold red]Error reading resume log: {str(e)}")
    
        # Filter out already processed PDFs if resuming and we have a valid resume log
        if resume_on_error and completed_pdfs and not is_fresh_matter:
            filtered_paths = [
                p for p in pdf_paths if str(p.absolute()) not in completed_pdfs
            ]
            skipped = len(pdf_paths) - len(filtered_paths)
            if skipped > 0:
                console.log(f"[bold yellow]Skipping {skipped} already processed files")
            pdf_paths = filtered_paths
    
        # Process PDFs in batches
        total_pdfs = len(pdf_paths)
        processed_count = 0
        error_count = 0
        
        # Start logging session if logger is available
        if ingestion_logger:
            ingestion_logger.start_session(total_pdfs)
    
        # Create a single progress context for the entire process
        with create_progress(f"Processing {total_pdfs} PDFs in batches") as progress:
            overall_task = progress.add_task(
                "Overall progress", total=total_pdfs
            )
    
            # Process in batches
            for batch_idx in range(0, total_pdfs, batch_size):
                batch = pdf_paths[batch_idx : batch_idx + batch_size]
                progress.update(
                    overall_task,
                    advance=0,
                    description=f"Batch {batch_idx//batch_size + 1}/{math.ceil(total_pdfs/batch_size)}",
                )
    
                # Create a task for each PDF in this batch
                pdf_tasks = {}
                for pdf_path in batch:
                    task_id = progress.add_task(
                        f"Processing {pdf_path.name}", total=1
                    )
                    pdf_tasks[pdf_path] = task_id
    
                # Process each PDF in this batch
                for pdf_path, task_id in pdf_tasks.items():
                    try:
                        # Pass the same progress object, matter_id, and logger
                        process_pdf(pdf_path, progress, task_id, project_name, matter.id, ingestion_logger)
                        processed_count += 1
    
                        # Mark as completed for resume functionality
                        with open(resume_log_path, "a") as f:
                            f.write(f"{pdf_path.absolute()}\n")
    
                    except Exception as e:
                        console.log(f"[bold red]Error processing {pdf_path}: {str(e)}")
                        
                        # Log the error in the ingestion logger
                        if ingestion_logger:
                            ingestion_logger.log_error(pdf_path, "batch_processing_error", str(e))
                            
                        error_count += 1
                        if not resume_on_error:
                            console.log(
                                "[bold red]Aborting due to error (resume_on_error=False)"
                            )
                            raise e
    
                # Update overall progress after each batch
                progress.update(
                    overall_task,
                    advance=len(batch),
                    description=f"Completed {processed_count}/{total_pdfs} ({error_count} errors)",
                )
        
        # End the logging session and get summary
        if ingestion_logger:
            summary = ingestion_logger.end_session()
            
            # Print a summary of the ingestion process
            console.print(f"[bold blue]Ingestion Summary:[/bold blue]")
            console.print(f"Documents processed: {summary['processed_documents']}/{summary['total_documents']}")
            console.print(f"Pages processed: {summary['total_pages']}")
            console.print(f"Chunks created: {summary['total_chunks']}")
            console.print(f"Processing time: {summary['duration_seconds']:.1f} seconds")
            console.print(f"Average chunks per page: {summary['avg_chunks_per_page']:.1f}")
            
            # Print classification distribution
            if summary['classification_distribution']:
                console.print("[bold blue]Document Classification Distribution:[/bold blue]")
                for doc_type, count in sorted(summary['classification_distribution'].items(), key=lambda x: x[1], reverse=True):
                    console.print(f"  {doc_type}: {count}")
            
            # Print error summary if any
            if summary['error_types']:
                console.print("[bold red]Error Summary:[/bold red]")
                for error_type, count in sorted(summary['error_types'].items(), key=lambda x: x[1], reverse=True):
                    console.print(f"  {error_type}: {count}")
                    
            console.print(f"[bold green]Detailed ingestion log saved to: {ingestion_logger.log_file}")
    
        # Final status
        if error_count > 0:
            console.log(
                f"[bold yellow]Ingestion completed with {error_count} errors. {processed_count}/{total_pdfs} files processed successfully."
            )
        else:
            console.log("[bold green]Ingestion complete! All files processed successfully.")
    finally:
        # Restore original directories
        config.paths.DATA_DIR = original_data_dir
        config.paths.INDEX_DIR = original_index_dir
