"""PDF ingestion module for claim-assistant."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import faiss
import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from rich.progress import Progress, TaskID

from .config import get_config, ensure_dirs
from .database import (
    init_database,
    is_page_processed,
    save_page_chunk,
    get_database_engine,
)
from .utils import (
    calculate_sha256,
    get_page_hash,
    process_page,
    create_progress,
    console,
)


# Chunk type classification prompt
CHUNK_TYPE_PROMPT = """
You are a specialized construction document analyzer for construction claims. Classify this document chunk into one of the following categories and include a confidence score (0-100%):

- Email: Any email correspondence between project stakeholders
- ChangeOrder: Documents describing changes to project scope, timeline, or costs
- Invoice: Bills, receipts, financial documents showing payments or charges
- Photo: Document is primarily a photograph with minimal text (construction site images, damage photos)
- ContractClause: Excerpts from the contract, specifications, or legal agreements
- Schedule: Project schedules, timelines, Gantt charts, or delay analysis
- DailyReport: Daily work reports, progress logs, or site condition documentation
- Drawing: Technical drawings, blueprints, or design documents
- Submittal: Material or equipment submittals and approvals
- RFI: Request for Information documents between contractors and designers
- Claim: Formal claim documents, dispute notices, or entitlement analyses
- Other: Anything that doesn't fit the above categories

Document text:
{text}

FORMAT YOUR RESPONSE AS:
Category: [category name]
Confidence: [0-100%]
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


def extract_dates(text: str) -> List[str]:
    """Extract dates from text using regex patterns."""
    dates = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates


def extract_document_ids(text: str) -> List[str]:
    """Extract document IDs from text using regex patterns."""
    ids = []
    for pattern in ID_PATTERNS:
        matches = re.findall(pattern, text)
        ids.extend(matches)
    return ids


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
            model=config.openai.MODEL,
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
            temperature=0.0,
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
    dim: int = 1536,
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
            # Determine embedding dimension (usually 1536 for text-embedding-3-large)
            embedding_dim = 1536
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


def process_pdf(pdf_path: Path, progress: Progress, task_id: TaskID) -> None:
    """Process a PDF file: extract text, metadata, generate embeddings."""
    config = get_config()
    console.log(f"Processing {pdf_path}")

    # Initialize document
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        console.log(f"[bold red]Error opening {pdf_path}: {str(e)}")
        progress.update(task_id, advance=1, status=f"Error: {str(e)}")
        return

    # Get/create FAISS index
    index = create_or_load_faiss_index()

    # Process each page
    total_pages = len(doc)
    progress.update(task_id, total=total_pages, status="Starting")

    for page_num in range(total_pages):
        progress.update(task_id, advance=0, status=f"Page {page_num+1}/{total_pages}")

        # Generate unique hash for this page
        page_hash = get_page_hash(pdf_path, page_num)

        # Skip if already processed
        if is_page_processed(page_hash):
            console.log(f"Skipping page {page_num+1} (already processed)")
            progress.update(task_id, advance=1, status=f"Skipped (duplicate)")
            continue

        # Process page
        try:
            page_data = process_page(doc, page_num)
            text = page_data["text"]

            # Extract metadata
            dates = extract_dates(text)
            doc_ids = extract_document_ids(text)

            # Classify chunk
            chunk_type, confidence = classify_chunk(text)

            # Create chunks for better semantic search
            chunk_size = config.chunking.CHUNK_SIZE
            chunk_overlap = config.chunking.CHUNK_OVERLAP

            # Only chunk text if it exceeds the minimum length
            text_chunks = chunk_text(text, chunk_size, chunk_overlap)
            console.log(f"Created {len(text_chunks)} chunks from page {page_num+1}")

            # Process each chunk
            for i, chunk_text in enumerate(text_chunks):
                # Create unique ID for this chunk
                chunk_id = f"{page_hash}_{i}"

                # Prepare DB entry
                chunk_data = {
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "page_num": page_num + 1,  # 1-indexed for humans
                    "page_hash": page_hash,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "image_path": page_data["image_path"],
                    "text": chunk_text,
                    "chunk_type": chunk_type,
                    "confidence": confidence,
                    "doc_date": dates[0] if dates else None,
                    "doc_id": doc_ids[0] if doc_ids else None,
                }

                # Generate embedding
                embedding = get_embeddings([chunk_text])[0]

                # Add to FAISS index
                index.add(np.array([embedding], dtype=np.float32))

                # Save to database
                save_page_chunk(chunk_data)

            progress.update(
                task_id, advance=1, status=f"Processed {chunk_type} ({confidence}%)"
            )

        except Exception as e:
            console.log(f"[bold red]Error processing page {page_num+1}: {str(e)}")
            progress.update(task_id, advance=1, status=f"Error: {str(e)}")

    # Close document
    doc.close()

    # Save updated index
    save_faiss_index(index)


def ingest_pdfs(pdf_paths: List[Path]) -> None:
    """Ingest a list of PDF files."""
    # Ensure directories exist
    ensure_dirs()

    # Initialize database
    init_database()

    # Create progress context
    with create_progress("Ingesting PDFs") as progress:
        # Create a task for each PDF
        pdf_tasks = {}
        for pdf_path in pdf_paths:
            task_id = progress.add_task(
                f"Processing {pdf_path.name}", total=1, status="Waiting"
            )
            pdf_tasks[pdf_path] = task_id

        # Process each PDF
        for pdf_path, task_id in pdf_tasks.items():
            process_pdf(pdf_path, progress, task_id)

    console.log("[bold green]Ingestion complete!")
