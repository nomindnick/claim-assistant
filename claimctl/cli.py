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
        True,
        "--resume/--no-resume",
        help="Resume processing from previous run if it was interrupted",
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
            default=True,
        )
        if not confirm:
            console.print("[bold yellow]Ingestion cancelled")
            raise typer.Exit(0)

    try:
        ingest_pdfs(
            expanded_paths,
            project_name=project,
            batch_size=batch_size,
            resume_on_error=resume,
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


@app.command("clear")
def clear_command(
    database: bool = typer.Option(
        False, "--database", "-d", help="Clear database records"
    ),
    embeddings: bool = typer.Option(
        False, "--embeddings", "-e", help="Clear FAISS embeddings"
    ),
    images: bool = typer.Option(
        False, "--images", "-i", help="Clear images and thumbnails"
    ),
    cache: bool = typer.Option(
        False, "--cache", "-c", help="Clear cache files and temporary data"
    ),
    exhibits: bool = typer.Option(
        False, "--exhibits", "-x", help="Clear exported exhibits"
    ),
    exports: bool = typer.Option(
        False, "--exports", "-p", help="Clear exported response PDFs"
    ),
    resume_log: bool = typer.Option(
        False, "--resume-log", "-r", help="Clear resume log file"
    ),
    all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Clear everything (database, embeddings, images, cache, exhibits, exports, resume log)",
    ),
) -> None:
    """Clear data from previous runs to start fresh."""
    config = get_config()

    # If no specific option is selected, show help
    if not (database or embeddings or images or cache or exhibits or exports or resume_log or all):
        console.print(
            "[bold yellow]No clear options selected. Please specify what to clear."
        )
        console.print(
            "Options: --database, --embeddings, --images, --cache, --exhibits, --exports, --resume-log, or --all"
        )
        return

    # If --all is specified, set all options to True
    if all:
        database = embeddings = images = cache = exhibits = exports = resume_log = True

    # Ask for confirmation
    should_clear = typer.confirm(
        "Are you sure you want to clear selected data? This cannot be undone."
    )
    if not should_clear:
        console.print("[bold yellow]Operation cancelled.")
        return

    deleted_count = 0

    # Clear database
    if database:
        try:
            db_path = Path(config.paths.INDEX_DIR) / "catalog.db"
            if db_path.exists():
                db_path.unlink()
                console.print(f"[bold green]Database cleared: {db_path}")
                deleted_count += 1
            else:
                console.print(f"[bold yellow]Database not found: {db_path}")
        except Exception as e:
            console.print(f"[bold red]Error clearing database: {str(e)}")

    # Clear embeddings
    if embeddings:
        try:
            index_path = Path(config.paths.INDEX_DIR) / "faiss.idx"
            if index_path.exists():
                index_path.unlink()
                console.print(f"[bold green]Embeddings cleared: {index_path}")
                deleted_count += 1
            else:
                console.print(f"[bold yellow]Embeddings index not found: {index_path}")
        except Exception as e:
            console.print(f"[bold red]Error clearing embeddings: {str(e)}")

    # Clear images and thumbnails
    if images:
        try:
            # Clear full-size images
            pages_dir = Path(config.paths.DATA_DIR) / "pages"
            if pages_dir.exists():
                image_count = 0
                for img_file in pages_dir.glob("*.png"):
                    img_file.unlink()
                    image_count += 1
                console.print(
                    f"[bold green]Cleared {image_count} images from {pages_dir}"
                )
                deleted_count += image_count

            # Clear thumbnails
            thumbs_dir = Path(config.paths.DATA_DIR) / "thumbnails"
            if thumbs_dir.exists():
                thumb_count = 0
                for thumb_file in thumbs_dir.glob("*.png"):
                    thumb_file.unlink()
                    thumb_count += 1
                console.print(
                    f"[bold green]Cleared {thumb_count} thumbnails from {thumbs_dir}"
                )
                deleted_count += thumb_count
        except Exception as e:
            console.print(f"[bold red]Error clearing images: {str(e)}")

    # Clear cache files
    if cache:
        try:
            cache_dir = Path(config.paths.DATA_DIR) / "cache"
            if cache_dir.exists():
                cache_count = 0
                for cache_file in cache_dir.glob("*.*"):
                    cache_file.unlink()
                    cache_count += 1
                console.print(f"[bold green]Cleared {cache_count} files from cache")
                deleted_count += cache_count
            else:
                console.print(f"[bold yellow]Cache directory not found: {cache_dir}")
        except Exception as e:
            console.print(f"[bold red]Error clearing cache: {str(e)}")

    # Clear exhibits
    if exhibits:
        try:
            exhibits_dir = Path("./exhibits")
            if exhibits_dir.exists():
                exhibits_count = 0
                for exhibit_file in exhibits_dir.glob("*.png"):
                    exhibit_file.unlink()
                    exhibits_count += 1
                console.print(f"[bold green]Cleared {exhibits_count} exported exhibits")
                deleted_count += exhibits_count
            else:
                console.print(
                    f"[bold yellow]Exhibits directory not found: {exhibits_dir}"
                )
        except Exception as e:
            console.print(f"[bold red]Error clearing exhibits: {str(e)}")
            
    # Clear exports
    if exports:
        try:
            exports_dir = Path("./Ask_Exports")
            if exports_dir.exists():
                exports_count = 0
                for export_file in exports_dir.glob("*.pdf"):
                    export_file.unlink()
                    exports_count += 1
                console.print(f"[bold green]Cleared {exports_count} exported response PDFs")
                deleted_count += exports_count
            else:
                console.print(
                    f"[bold yellow]Exports directory not found: {exports_dir}"
                )
        except Exception as e:
            console.print(f"[bold red]Error clearing exports: {str(e)}")

    # Clear resume log
    if resume_log:
        try:
            resume_log_path = Path(config.paths.DATA_DIR) / "ingest_resume.log"
            if resume_log_path.exists():
                resume_log_path.unlink()
                console.print(f"[bold green]Resume log cleared: {resume_log_path}")
                deleted_count += 1
            else:
                console.print(f"[bold yellow]Resume log not found: {resume_log_path}")
        except Exception as e:
            console.print(f"[bold red]Error clearing resume log: {str(e)}")

    # Final summary
    if deleted_count > 0:
        console.print(f"[bold green]Successfully cleared {deleted_count} items")
    else:
        console.print("[bold yellow]No items were cleared")


@app.command("version")
def version_command() -> None:
    """Show version information."""
    console.print(f"claim-assistant v{__version__}")


# Add the commands to the app
if __name__ == "__main__":
    app()
