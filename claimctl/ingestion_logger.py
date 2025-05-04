"""Ingestion logger for tracking document processing metrics and errors."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console

console = Console()


class IngestionLogger:
    """Logger for tracking document ingestion metrics, errors, and statistics."""

    def __init__(self, matter_name: str, matter_dir: Path):
        """Initialize the ingestion logger for a specific matter.
        
        Args:
            matter_name: Name of the matter being processed
            matter_dir: Path to the matter directory
        """
        self.matter_name = matter_name
        self.logs_dir = matter_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"ingestion_{timestamp}.jsonl"
        
        # Session metrics
        self.session_start_time = time.time()
        self.session_end_time = None
        self.total_docs = 0
        self.processed_docs = 0
        self.error_docs = 0
        self.total_pages = 0
        self.total_chunks = 0
        self.classification_counts: Dict[str, int] = {}
        self.extraction_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Create logger
        self.logger = logging.getLogger(f"ingestion.{matter_name}")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        
        # Log session start
        self._log_event("session_start", {
            "matter_name": matter_name,
            "timestamp": datetime.now().isoformat(),
            "log_file": str(self.log_file)
        })
        
        console.print(f"[green]Ingestion logging started: {self.log_file}[/green]")
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to the log file.
        
        Args:
            event_type: Type of event being logged
            data: Event data to log
        """
        log_entry = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to write to ingestion log: {e}[/yellow]")
    
    def start_session(self, pdf_count: int) -> None:
        """Log the start of an ingestion session.
        
        Args:
            pdf_count: Number of PDFs to be processed
        """
        self.total_docs = pdf_count
        self.session_start_time = time.time()
        
        self._log_event("session_info", {
            "action": "start",
            "pdf_count": pdf_count,
            "start_time": datetime.now().isoformat()
        })
    
    def end_session(self) -> Dict[str, Any]:
        """Log the end of an ingestion session and generate a summary.
        
        Returns:
            Summary statistics for the session
        """
        self.session_end_time = time.time()
        duration = self.session_end_time - self.session_start_time
        
        summary = {
            "total_documents": self.total_docs,
            "processed_documents": self.processed_docs,
            "error_documents": self.error_docs,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "duration_seconds": round(duration, 2),
            "classification_distribution": self.classification_counts,
            "extraction_counts": self.extraction_counts,
            "error_types": self.error_counts,
            "avg_chunks_per_page": round(self.total_chunks / max(1, self.total_pages), 2),
            "avg_pages_per_document": round(self.total_pages / max(1, self.processed_docs), 2),
            "error_rate": round(self.error_docs / max(1, self.total_docs) * 100, 2)
        }
        
        self._log_event("session_summary", summary)
        
        # Log session end
        self._log_event("session_info", {
            "action": "end",
            "processed_count": self.processed_docs,
            "error_count": self.error_docs,
            "end_time": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2)
        })
        
        return summary
    
    def log_document_start(self, pdf_path: Union[str, Path]) -> None:
        """Log the start of document processing.
        
        Args:
            pdf_path: Path to the PDF being processed
        """
        self._log_event("document_processing", {
            "action": "start",
            "pdf_path": str(pdf_path),
            "filename": os.path.basename(str(pdf_path))
        })
    
    def log_document_complete(self, 
                              pdf_path: Union[str, Path], 
                              processing_time: float, 
                              page_count: int, 
                              chunk_count: int,
                              success: bool = True) -> None:
        """Log the completion of document processing.
        
        Args:
            pdf_path: Path to the PDF that was processed
            processing_time: Time taken to process the document in seconds
            page_count: Number of pages in the document
            chunk_count: Number of chunks created from the document
            success: Whether processing was successful
        """
        if success:
            self.processed_docs += 1
            self.total_pages += page_count
            self.total_chunks += chunk_count
        else:
            self.error_docs += 1
        
        self._log_event("document_processing", {
            "action": "complete",
            "pdf_path": str(pdf_path),
            "filename": os.path.basename(str(pdf_path)),
            "processing_time_seconds": round(processing_time, 2),
            "page_count": page_count,
            "chunk_count": chunk_count,
            "success": success
        })
    
    def log_classification(self, 
                          pdf_path: Union[str, Path], 
                          page_num: int, 
                          chunk_index: int,
                          chunk_type: str, 
                          confidence: int) -> None:
        """Log a chunk classification decision.
        
        Args:
            pdf_path: Path to the PDF containing the chunk
            page_num: Page number containing the chunk
            chunk_index: Index of the chunk on the page
            chunk_type: Classification of the chunk
            confidence: Confidence score (0-100) for the classification
        """
        # Update classification counts
        self.classification_counts[chunk_type] = self.classification_counts.get(chunk_type, 0) + 1
        
        self._log_event("classification", {
            "pdf_path": str(pdf_path),
            "filename": os.path.basename(str(pdf_path)),
            "page_num": page_num,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "confidence": confidence
        })
    
    def log_extraction(self, 
                      pdf_path: Union[str, Path],
                      page_num: int,
                      metadata_type: str,
                      extracted_value: Optional[str],
                      success: bool = True) -> None:
        """Log metadata extraction results.
        
        Args:
            pdf_path: Path to the PDF being processed
            page_num: Page number being processed
            metadata_type: Type of metadata extracted (e.g., "date", "amount")
            extracted_value: The extracted value
            success: Whether extraction was successful
        """
        # Update extraction counts
        key = f"{metadata_type}_{'success' if success else 'failure'}"
        self.extraction_counts[key] = self.extraction_counts.get(key, 0) + 1
        
        self._log_event("extraction", {
            "pdf_path": str(pdf_path),
            "filename": os.path.basename(str(pdf_path)),
            "page_num": page_num,
            "metadata_type": metadata_type,
            "extracted_value": extracted_value,
            "success": success
        })
    
    def log_error(self, 
                 pdf_path: Union[str, Path], 
                 error_type: str, 
                 error_message: str,
                 page_num: Optional[int] = None) -> None:
        """Log a processing error.
        
        Args:
            pdf_path: Path to the PDF that caused the error
            error_type: Type of error that occurred
            error_message: Detailed error message
            page_num: Optional page number where the error occurred
        """
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self._log_event("error", {
            "pdf_path": str(pdf_path),
            "filename": os.path.basename(str(pdf_path)),
            "page_num": page_num,
            "error_type": error_type,
            "error_message": error_message
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get the current session summary statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        duration = time.time() - self.session_start_time
        
        return {
            "total_documents": self.total_docs,
            "processed_documents": self.processed_docs,
            "error_documents": self.error_docs,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "duration_seconds": round(duration, 2),
            "classification_distribution": self.classification_counts,
            "extraction_counts": self.extraction_counts,
            "error_types": self.error_counts,
            "avg_chunks_per_page": round(self.total_chunks / max(1, self.total_pages), 2),
            "avg_pages_per_document": round(self.total_pages / max(1, self.processed_docs), 2),
            "error_rate": round(self.error_docs / max(1, self.total_docs) * 100, 2)
        }


def analyze_log_file(log_file: Path) -> Dict[str, Any]:
    """Analyze an ingestion log file and generate statistics.
    
    Args:
        log_file: Path to the log file to analyze
        
    Returns:
        Dictionary of analysis results
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    events = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                events.append(event)
            except json.JSONDecodeError:
                continue
    
    # Extract summary statistics if available
    for event in events:
        if event.get("event") == "session_summary":
            # Return the pre-calculated summary if available
            return {k: v for k, v in event.items() if k != "event"}
    
    # If no summary is available, calculate one
    start_time = None
    end_time = None
    total_docs = 0
    processed_docs = 0
    error_docs = 0
    classification_counts = {}
    error_counts = {}
    
    for event in events:
        if event.get("event") == "session_info":
            if event.get("action") == "start":
                start_time = event.get("timestamp")
                total_docs = event.get("pdf_count", 0)
            elif event.get("action") == "end":
                end_time = event.get("timestamp")
                processed_docs = event.get("processed_count", 0)
                error_docs = event.get("error_count", 0)
        
        elif event.get("event") == "classification":
            chunk_type = event.get("chunk_type", "unknown")
            classification_counts[chunk_type] = classification_counts.get(chunk_type, 0) + 1
        
        elif event.get("event") == "error":
            error_type = event.get("error_type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    return {
        "log_file": str(log_file),
        "start_time": start_time,
        "end_time": end_time,
        "total_documents": total_docs,
        "processed_documents": processed_docs,
        "error_documents": error_docs,
        "classification_distribution": classification_counts,
        "error_types": error_counts
    }


def get_recent_logs(matter_dir: Path, limit: int = 5) -> List[Path]:
    """Get the most recent ingestion log files for a matter.
    
    Args:
        matter_dir: Path to the matter directory
        limit: Maximum number of log files to return
        
    Returns:
        List of paths to the most recent log files
    """
    logs_dir = matter_dir / "logs"
    if not logs_dir.exists():
        return []
    
    log_files = sorted(
        [f for f in logs_dir.glob("ingestion_*.jsonl")],
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    
    return log_files[:limit]