"""Command-line interface for claim-assistant."""

import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from . import __version__
from .config import ensure_dirs, get_config, show_config
from .ingest import ingest_pdfs
from .query import query_documents
from .utils import console

# Create Typer app
app = typer.Typer(
    name="claimctl",
    help="Construction Claim Assistant CLI",
    add_completion=False,
)


@app.command("ingest")
def ingest_command(
    pdf_paths: List[Path] = typer.Argument(
        ...,
        help="Path(s) to PDF files to ingest",
        exists=True,
        dir_okay=True,  # Allow directories to support batch processing
        resolve_path=True,
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project name to associate with documents"
    ),
    batch_size: int = typer.Option(
        5, "--batch-size", "-b", help="Number of PDFs to process in each batch"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume processing from previous run if it was interrupted"
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Recursively process directories"
    ),
) -> None:
    """Ingest PDF files into the claim assistant database."""
    # Expand directories to individual PDF files if needed
    expanded_paths = []
    
    for path in pdf_paths:
        if path.is_dir():
            # Find all PDFs in the directory
            if recursive:
                # Find PDFs recursively in all subdirectories
                for pdf_path in path.glob("**/*.pdf"):
                    expanded_paths.append(pdf_path)
            else:
                # Find PDFs only in the top directory
                for pdf_path in path.glob("*.pdf"):
                    expanded_paths.append(pdf_path)
        else:
            # Individual file (verify it's a PDF)
            if path.suffix.lower() != ".pdf":
                console.print(f"[bold red]Error: {path} is not a PDF file")
                raise typer.Exit(1)
            expanded_paths.append(path)
    
    # Sort paths for consistent processing order
    expanded_paths.sort()
    
    # Check if we found any PDFs
    if not expanded_paths:
        console.print("[bold red]Error: No PDF files found")
        raise typer.Exit(1)
        
    console.print(f"[bold green]Found {len(expanded_paths)} PDF files to process")
    
    # Confirm if large number of PDFs
    if len(expanded_paths) > 10:
        confirm = typer.confirm(
            f"Are you sure you want to process {len(expanded_paths)} PDF files?", 
            default=True
        )
        if not confirm:
            console.print("[bold yellow]Ingestion cancelled")
            raise typer.Exit(0)
    
    try:
        ingest_pdfs(
            expanded_paths, 
            project_name=project, 
            batch_size=batch_size, 
            resume_on_error=resume
        )
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")
        raise typer.Exit(1)


@app.command("ask")
def ask_command(
    question: str = typer.Argument(
        ..., help="Question to ask about the construction claim"
    ),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="Number of most relevant documents to consider"
    ),
    json: bool = typer.Option(False, "--json", help="Output results in JSON format"),
    markdown: bool = typer.Option(
        False, "--md", help="Output results in Markdown format"
    ),
    doc_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by document type"
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project name"
    ),
    date_from: Optional[str] = typer.Option(
        None, "--from", help="Filter documents from this date (YYYY-MM-DD)"
    ),
    date_to: Optional[str] = typer.Option(
        None, "--to", help="Filter documents to this date (YYYY-MM-DD)"
    ),
    parties: Optional[str] = typer.Option(
        None, "--parties", help="Filter by parties involved"
    ),
    search_type: str = typer.Option(
        "hybrid", "--search", "-s", help="Search type: hybrid, vector, or keyword"
    ),
) -> None:
    """Ask a question about the construction claim."""
    try:
        query_documents(
            question,
            top_k,
            json,
            markdown,
            doc_type,
            date_from,
            date_to,
            project,
            parties,
            search_type,
        )
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")
        raise typer.Exit(1)


@app.command("config")
def config_command(
    action: str = typer.Argument(..., help="Action to perform: show, init"),
) -> None:
    """View or modify configuration."""
    if action.lower() == "show":
        config = show_config()
        console.print(config)
    elif action.lower() == "init":
        ensure_dirs()
        console.print("[bold green]Directories initialized!")
    else:
        console.print(f"[bold red]Unknown action: {action}")
        console.print("Available actions: show, init")
        raise typer.Exit(1)


@app.command("version")
def version_command() -> None:
    """Show version information."""
    console.print(f"claim-assistant v{__version__}")


# Add the commands to the app
if __name__ == "__main__":
    app()
