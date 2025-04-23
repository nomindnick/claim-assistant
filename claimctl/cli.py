"""Command-line interface for claim-assistant."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from rich.console import Console

from . import __version__
from .config import get_config, ensure_dirs, show_config
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
        dir_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Ingest PDF files into the claim assistant database."""
    # Ensure all files are PDFs
    for path in pdf_paths:
        if path.suffix.lower() != ".pdf":
            console.print(f"[bold red]Error: {path} is not a PDF file")
            raise typer.Exit(1)

    try:
        ingest_pdfs(pdf_paths)
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
) -> None:
    """Ask a question about the construction claim."""
    try:
        query_documents(question, top_k, json, markdown)
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
