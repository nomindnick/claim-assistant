"""Command-line interface for claim-assistant."""

import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import (
    ensure_dirs, 
    get_config, 
    show_config, 
    get_current_matter, 
    set_current_matter, 
    get_matter_path
)
from .database import get_session, Matter, Document, init_database
from .ingest import ingest_pdfs
from .query import query_documents
from .timeline import (
    EventType,
    generate_claim_timeline,
    display_timeline,
    extract_events_from_all_documents
)
from .utils import console, export_timeline_as_pdf

# Create Typer app
app = typer.Typer(
    name="claimctl",
    help="Construction Claim Assistant CLI (use without arguments for interactive shell)",
    add_completion=False,
)

# Create subcommands
logs_app = typer.Typer(help="Manage and analyze ingestion logs")
app.add_typer(logs_app, name="logs")

timeline_app = typer.Typer(help="Generate and manage claim timelines")
app.add_typer(timeline_app, name="timeline")

financials_app = typer.Typer(help="Manage financial events in timelines")
timeline_app.add_typer(financials_app, name="financials")

contradictions_app = typer.Typer(help="Manage contradictions in timelines")
timeline_app.add_typer(contradictions_app, name="contradictions")


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
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to associate with documents"
    ),
    semantic_chunking: bool = typer.Option(
        None, "--semantic-chunking/--no-semantic-chunking", 
        help="Use semantic chunking instead of character-based chunking"
    ),
    hierarchical_chunking: bool = typer.Option(
        None, "--hierarchical-chunking/--no-hierarchical-chunking", 
        help="Use hierarchical chunking for structured documents"
    ),
    adaptive_chunking: bool = typer.Option(
        None, "--adaptive-chunking/--no-adaptive-chunking", 
        help="Automatically detect document structure and choose optimal chunking method"
    ),
    logging: bool = typer.Option(
        True, "--logging/--no-logging", 
        help="Enable detailed ingestion logging"
    ),
    timeline_extract: bool = typer.Option(
        None, "--timeline-extract/--no-timeline-extract",
        help="Enable automatic timeline event extraction during ingestion"
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
        # Get config to update chunking options
        config = get_config()
        
        # Update config with command-line options if provided
        if semantic_chunking is not None:
            config.chunking.SEMANTIC_CHUNKING = semantic_chunking
        if hierarchical_chunking is not None:
            config.chunking.HIERARCHICAL_CHUNKING = hierarchical_chunking
        if adaptive_chunking is not None:
            config.chunking.ADAPTIVE_CHUNKING = adaptive_chunking
        if timeline_extract is not None:
            config.timeline.AUTO_EXTRACT = timeline_extract
        
        ingest_pdfs(
            expanded_paths,
            project_name=project,
            batch_size=batch_size,
            resume_on_error=resume,
            matter_name=matter,  # Pass matter name
            enable_logging=logging,  # Pass logging flag
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
    amount_min: Optional[float] = typer.Option(
        None, "--amount-min", help="Minimum monetary amount"
    ),
    amount_max: Optional[float] = typer.Option(
        None, "--amount-max", help="Maximum monetary amount"
    ),
    section: Optional[str] = typer.Option(
        None, "--section", help="Filter by contract section reference"
    ),
    public_agency: bool = typer.Option(
        False, "--public-agency", help="Only return public agency documents"
    ),
    search_type: str = typer.Option(
        "hybrid", "--search", "-s", help="Search type: hybrid, vector, or keyword"
    ),
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter to query"
    ),
) -> None:
    """Ask a question about the construction claim."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    try:
        console.log(f"CLI: Calling query_documents with top_k={top_k}")
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
            amount_min=amount_min,
            amount_max=amount_max,
            section_reference=section,
            public_agency=public_agency,
            search_type=search_type,
            matter=matter,  # Pass matter parameter
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
        False, "--exports", "-p", help="Clear exported PDFs (responses and timelines)"
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
                # Check if current matter exists before deleting
                current_matter = get_current_matter()
                
                # Delete database file
                db_path.unlink()
                console.print(f"[bold green]Database cleared: {db_path}")
                
                # Re-initialize database after deletion
                init_database()
                console.print("[bold green]Database schema re-initialized")
                
                # Also clear all resume log files to force reprocessing of PDFs
                # Clear main resume log
                resume_log_path = Path(config.paths.DATA_DIR) / "ingest_resume.log"
                if resume_log_path.exists():
                    resume_log_path.unlink()
                    console.print(f"[bold green]Resume log cleared: {resume_log_path}")
                
                # Also look for matter-specific logs in matter directories
                try:
                    matter_dirs = Path("./matters").glob("*")
                    for matter_dir in matter_dirs:
                        if matter_dir.is_dir():
                            matter_resume_log = matter_dir / "data" / "ingest_resume.log"
                            if matter_resume_log.exists():
                                matter_resume_log.unlink()
                                console.print(f"[bold green]Matter resume log cleared: {matter_resume_log}")
                except Exception as e:
                    console.print(f"[bold yellow]Note: Error clearing matter resume logs: {e}")
                
                # Reset current matter in config since it no longer exists
                if current_matter:
                    from .config import set_current_matter
                    set_current_matter("")
                    console.print(f"[bold yellow]Active matter '{current_matter}' was cleared. No active matter.")
                
                deleted_count += 1
            else:
                console.print(f"[bold yellow]Database not found: {db_path}")
                # Initialize the database if it doesn't exist
                init_database()
                console.print("[bold green]Database schema initialized")
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
            # Clear Ask_Exports directory
            ask_exports_dir = Path("./Ask_Exports")
            exports_count = 0
            if ask_exports_dir.exists():
                for export_file in ask_exports_dir.glob("*.pdf"):
                    export_file.unlink()
                    exports_count += 1
                console.print(f"[bold green]Cleared {exports_count} exported response PDFs")
                deleted_count += exports_count
            else:
                console.print(
                    f"[bold yellow]Ask_Exports directory not found: {ask_exports_dir}"
                )
                
            # Clear Timeline_Exports directory
            timeline_exports_dir = Path("./Timeline_Exports")
            timeline_exports_count = 0
            if timeline_exports_dir.exists():
                for export_file in timeline_exports_dir.glob("*.pdf"):
                    export_file.unlink()
                    timeline_exports_count += 1
                console.print(f"[bold green]Cleared {timeline_exports_count} exported timeline PDFs")
                deleted_count += timeline_exports_count
            else:
                console.print(
                    f"[bold yellow]Timeline_Exports directory not found: {timeline_exports_dir}"
                )
        except Exception as e:
            console.print(f"[bold red]Error clearing exports: {str(e)}")

    # Clear resume log
    if resume_log:
        try:
            # Clear global resume log
            resume_log_path = Path(config.paths.DATA_DIR) / "ingest_resume.log"
            if resume_log_path.exists():
                resume_log_path.unlink()
                console.print(f"[bold green]Resume log cleared: {resume_log_path}")
                deleted_count += 1
            else:
                console.print(f"[bold yellow]Resume log not found: {resume_log_path}")
                
            # Also clear matter-specific resume logs
            from .database import get_session, Matter
            try:
                with get_session() as session:
                    matters = session.query(Matter).all()
                    for matter in matters:
                        matter_resume_log = Path(matter.data_directory) / "ingest_resume.log"
                        if matter_resume_log.exists():
                            matter_resume_log.unlink()
                            console.print(f"[bold green]Matter resume log cleared: {matter_resume_log}")
                            deleted_count += 1
            except Exception as matter_error:
                console.print(f"[bold yellow]Note: Could not clear matter resume logs: {matter_error}")
        except Exception as e:
            console.print(f"[bold red]Error clearing resume log: {str(e)}")

    # Final summary
    if deleted_count > 0:
        console.print(f"[bold green]Successfully cleared {deleted_count} items")
    else:
        console.print("[bold yellow]No items were cleared")


@app.command("matter")
def matter_command(
    action: str = typer.Argument(..., help="Action: list, create, switch, info, delete"),
    name: Optional[str] = typer.Argument(None, help="Matter name"),
    description: Optional[str] = typer.Option(None, "--desc", help="Matter description"),
) -> None:
    """Manage legal matters."""
    if action.lower() == "list":
        list_matters()
    elif action.lower() == "create":
        create_matter(name, description)
    elif action.lower() == "switch":
        switch_matter(name)
    elif action.lower() == "info":
        show_matter_info(name)
    elif action.lower() == "delete":
        delete_matter(name)
    else:
        console.print(f"[bold red]Unknown action: {action}")
        console.print("Available actions: list, create, switch, info, delete")
        raise typer.Exit(1)


def list_matters() -> None:
    """List all available matters."""
    with get_session() as session:
        matters = session.query(Matter).all()
        
        table = Table(title="Legal Matters")
        table.add_column("ID", justify="right", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        table.add_column("Documents", justify="right")
        table.add_column("Created", justify="right")
        table.add_column("Last Accessed", justify="right")
        table.add_column("Current", justify="center")
        
        current_matter = get_current_matter()
        
        for matter in matters:
            # Count documents - handle case where matter_id might be NULL in existing documents
            try:
                doc_count = session.query(Document).filter(Document.matter_id == matter.id).count()
            except Exception as e:
                console.print(f"[dim]Warning: {str(e)}[/dim]")
                doc_count = 0
            
            # Format date
            created_at = matter.created_at.strftime("%Y-%m-%d")
            last_accessed = matter.last_accessed.strftime("%Y-%m-%d")
            
            # Highlight current matter
            is_current = "✓" if matter.name == current_matter else ""
            
            table.add_row(
                str(matter.id),
                matter.name,
                matter.description or "",
                str(doc_count),
                created_at,
                last_accessed,
                is_current
            )
        
        console.print(table)


def create_matter(name: str, description: Optional[str] = None) -> None:
    """Create a new matter."""
    if not name:
        console.print("[bold red]Error: Matter name is required")
        raise typer.Exit(1)
        
    # Initialize database
    init_database()
    
    # Create matter directories
    matter_dir = get_matter_path(name)
    data_dir = matter_dir / "data"
    index_dir = matter_dir / "index"
    
    for directory in [matter_dir, data_dir, data_dir / "raw", data_dir / "pages", data_dir / "cache", index_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Create matter in database
    try:
        with get_session() as session:
            # Check if matter already exists
            existing = session.query(Matter).filter(Matter.name == name).first()
            if existing:
                console.print(f"[bold yellow]Matter '{name}' already exists")
                raise typer.Exit(1)
                
            # Create new matter
            matter = Matter(
                name=name,
                description=description,
                data_directory=str(data_dir),
                index_directory=str(index_dir)
            )
            session.add(matter)
            session.commit()
            
        console.print(f"[bold green]Matter '{name}' created successfully")
        
        # Ask if user wants to switch to new matter
        switch = typer.confirm(f"Switch to matter '{name}'?", default=True)
        if switch:
            switch_matter(name)
            
    except Exception as e:
        console.print(f"[bold red]Error creating matter: {str(e)}")
        raise typer.Exit(1)


def switch_matter(name: str) -> None:
    """Switch to a different matter."""
    if not name:
        console.print("[bold red]Error: Matter name is required")
        raise typer.Exit(1)
        
    try:
        with get_session() as session:
            # Find matter
            matter = session.query(Matter).filter(Matter.name == name).first()
            if not matter:
                console.print(f"[bold red]Matter '{name}' not found")
                raise typer.Exit(1)
                
            # Update last accessed timestamp
            matter.last_accessed = datetime.utcnow()
            session.commit()
            
            # Update current matter in config
            set_current_matter(name)
            
            console.print(f"[bold green]Switched to matter: {name}")
            
            # Show matter info
            show_matter_info(name)
            
    except Exception as e:
        console.print(f"[bold red]Error switching matter: {str(e)}")
        raise typer.Exit(1)


def show_matter_info(name: Optional[str] = None) -> None:
    """Show matter information."""
    # Use current matter if not specified
    if not name:
        name = get_current_matter()
        if not name:
            console.print("[bold yellow]No active matter")
            console.print("Use 'matter switch <n>' to select a matter")
            return
            
    try:
        with get_session() as session:
            matter = session.query(Matter).filter(Matter.name == name).first()
            if not matter:
                console.print(f"[bold red]Matter '{name}' not found")
                return
                
            # Count documents - handle case where matter_id might be NULL in existing documents
            try:
                doc_count = session.query(Document).filter(Document.matter_id == matter.id).count()
            except Exception as e:
                console.print(f"[dim]Warning: {str(e)}[/dim]")
                doc_count = 0
            
            # Display matter information
            console.print(f"[bold]Matter: {matter.name}")
            if matter.description:
                console.print(f"[bold]Description: {matter.description}")
            console.print(f"[bold]Documents: {doc_count}")
            console.print(f"[bold]Data Directory: {matter.data_directory}")
            console.print(f"[bold]Index Directory: {matter.index_directory}")
            console.print(f"[bold]Created: {matter.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"[bold]Last Accessed: {matter.last_accessed.strftime('%Y-%m-%d %H:%M:%S')}")
            
    except Exception as e:
        console.print(f"[bold red]Error showing matter info: {str(e)}")


def delete_matter(name: str) -> None:
    """Delete a matter."""
    if not name:
        console.print("[bold red]Error: Matter name is required")
        raise typer.Exit(1)
    
    # Check if this is the active matter
    current_matter = get_current_matter()
    if current_matter == name:
        # Check if this is the only matter
        with get_session() as session:
            matters_count = session.query(Matter).count()
            
        if matters_count <= 1:
            console.print("[bold red]Cannot delete the only matter")
            console.print("Please create another matter first with 'matter create <n>'")
            console.print("Then switch to that matter before deleting this one")
            raise typer.Exit(1)
        else:
            console.print("[bold red]Cannot delete the active matter")
            console.print("Please switch to a different matter first")
            raise typer.Exit(1)
        
    # Confirm deletion
    confirm = typer.confirm(f"Are you sure you want to delete matter '{name}'? This will remove all data.", default=False)
    if not confirm:
        console.print("[bold yellow]Operation cancelled")
        return
        
    try:
        with get_session() as session:
            # Find matter
            matter = session.query(Matter).filter(Matter.name == name).first()
            if not matter:
                console.print(f"[bold red]Matter '{name}' not found")
                raise typer.Exit(1)
                
            # Record paths to delete after database transaction
            data_dir = Path(matter.data_directory) if matter.data_directory else None
            index_dir = Path(matter.index_directory) if matter.index_directory else None
            
            # Delete matter from database (this will cascade to documents, pages, and chunks)
            session.delete(matter)
            session.commit()
            
            console.print(f"[bold green]Matter '{name}' deleted from database")
            
            # Ask if user wants to delete directories
            if data_dir or index_dir:
                delete_dirs = typer.confirm("Delete matter directories as well?", default=True)
                if delete_dirs:
                    import shutil
                    
                    # Delete directories if they exist
                    if data_dir and data_dir.exists():
                        shutil.rmtree(data_dir, ignore_errors=True)
                        console.print(f"[bold green]Deleted data directory: {data_dir}")
                        
                    if index_dir and index_dir.exists():
                        shutil.rmtree(index_dir, ignore_errors=True)
                        console.print(f"[bold green]Deleted index directory: {index_dir}")
                        
                    # Delete parent matter directory if empty
                    if data_dir and data_dir.parent.exists() and not any(data_dir.parent.iterdir()):
                        data_dir.parent.rmdir()
                        console.print(f"[bold green]Deleted empty matter directory: {data_dir.parent}")
            
    except Exception as e:
        console.print(f"[bold red]Error deleting matter: {str(e)}")
        raise typer.Exit(1)


@app.command("version")
def version_command() -> None:
    """Show version information."""
    console.print(f"claim-assistant v{__version__}")


@logs_app.command("list")
def logs_list_command(
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to list logs for"
    ),
    limit: int = typer.Option(
        5, "--limit", "-l", help="Maximum number of logs to list"
    ),
) -> None:
    """List recent ingestion logs for a matter."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter directory
    matter_dir = get_matter_path(matter)
    if not matter_dir:
        console.print(f"[bold red]Matter '{matter}' not found")
        raise typer.Exit(1)
    
    # Import here to avoid circular imports
    from .ingestion_logger import get_recent_logs
    
    # Get recent logs
    logs = get_recent_logs(matter_dir, limit)
    
    if not logs:
        console.print(f"[bold yellow]No ingestion logs found for matter '{matter}'")
        return
    
    # Create a table to display logs
    table = Table(title=f"Recent Ingestion Logs for '{matter}'")
    table.add_column("Date", style="cyan")
    table.add_column("Time", style="cyan")
    table.add_column("Log File", style="green")
    table.add_column("Size", justify="right", style="blue")
    
    for log_file in logs:
        # Extract timestamp from filename (format: ingestion_YYYYMMDD_HHMMSS.jsonl)
        filename = log_file.name
        if filename.startswith("ingestion_") and "_" in filename:
            timestamp = filename.split("_")[1]
            date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
            time_str = f"{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        else:
            # Use file modification time if filename doesn't contain timestamp
            mtime = log_file.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            date = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
        
        # Get file size in KB
        size_kb = log_file.stat().st_size / 1024
        
        table.add_row(
            date,
            time_str,
            log_file.name,
            f"{size_kb:.1f} KB"
        )
    
    console.print(table)


@logs_app.command("show")
def logs_show_command(
    log_file: Optional[str] = typer.Argument(
        None, help="Log file name to show (if omitted, shows most recent)"
    ),
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name the log belongs to"
    ),
) -> None:
    """Show summary of an ingestion log file."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter directory
    matter_dir = get_matter_path(matter)
    if not matter_dir:
        console.print(f"[bold red]Matter '{matter}' not found")
        raise typer.Exit(1)
    
    # Import here to avoid circular imports
    from .ingestion_logger import get_recent_logs, analyze_log_file
    
    # Get log file path
    log_path = None
    if log_file:
        # Use specified log file
        log_path = matter_dir / "logs" / log_file
        if not log_path.exists():
            console.print(f"[bold red]Log file '{log_file}' not found for matter '{matter}'")
            raise typer.Exit(1)
    else:
        # Use most recent log file
        recent_logs = get_recent_logs(matter_dir, 1)
        if not recent_logs:
            console.print(f"[bold yellow]No ingestion logs found for matter '{matter}'")
            raise typer.Exit(1)
        log_path = recent_logs[0]
    
    try:
        # Analyze log file
        summary = analyze_log_file(log_path)
        
        # Display summary
        console.print(f"[bold blue]Ingestion Log Summary: {log_path.name}[/bold blue]")
        console.print(f"Matter: [cyan]{matter}[/cyan]")
        
        if "start_time" in summary and summary["start_time"]:
            console.print(f"Start time: {summary['start_time']}")
        if "end_time" in summary and summary["end_time"]:
            console.print(f"End time: {summary['end_time']}")
        
        console.print(f"Documents processed: {summary.get('processed_documents', 0)}/{summary.get('total_documents', 0)}")
        
        if "total_pages" in summary:
            console.print(f"Pages processed: {summary['total_pages']}")
        if "total_chunks" in summary:
            console.print(f"Chunks created: {summary['total_chunks']}")
        if "duration_seconds" in summary:
            console.print(f"Processing time: {summary['duration_seconds']:.1f} seconds")
        if "avg_chunks_per_page" in summary:
            console.print(f"Average chunks per page: {summary['avg_chunks_per_page']:.1f}")
        
        # Print classification distribution
        if "classification_distribution" in summary and summary["classification_distribution"]:
            console.print("[bold blue]Document Classification Distribution:[/bold blue]")
            for doc_type, count in sorted(summary["classification_distribution"].items(), key=lambda x: x[1], reverse=True):
                console.print(f"  {doc_type}: {count}")
        
        # Print error summary if any
        if "error_types" in summary and summary["error_types"]:
            console.print("[bold red]Error Summary:[/bold red]")
            for error_type, count in sorted(summary["error_types"].items(), key=lambda x: x[1], reverse=True):
                console.print(f"  {error_type}: {count}")
            
            # Print error rate
            if "error_rate" in summary:
                console.print(f"Error rate: {summary['error_rate']:.1f}%")
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing log file: {str(e)}")
        raise typer.Exit(1)


@timeline_app.command("extract")
def timeline_extract_command(
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to extract timeline events for"
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Don't resume from previous extraction"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-extraction of all events"
    ),
    document_aware: bool = typer.Option(
        True, "--document-aware/--no-document-aware", help="Process chunks with document context"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel", help="Process documents in parallel"
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of parallel worker threads (1-8, default 4)", min=1, max=8
    ),
) -> None:
    """Extract timeline events from all documents in a matter.
    
    By default, the extraction will resume from where it left off if interrupted.
    Use --no-resume to start from the beginning, ignoring previous progress.
    Use --force to delete all existing timeline events and start fresh.
    Use --document-aware to process chunks with document context (default, recommended).
    Use --parallel to process documents in parallel for faster extraction (default).
    Use --workers to specify the number of parallel workers (default 4).
    """
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter ID
    with get_session() as session:
        matter_obj = session.query(Matter).filter(Matter.name == matter).first()
        if not matter_obj:
            console.print(f"[bold red]Matter '{matter}' not found")
            raise typer.Exit(1)
        
        matter_id = matter_obj.id
    
    # Show appropriate confirmation message based on options
    message = f"Extract timeline events from all documents in matter '{matter}'?"
    if force:
        message += " (force mode: will delete all existing timeline events)"
    elif not no_resume:
        message += " (will resume from previous extraction if interrupted)"
    else:
        message += " (will start from the beginning)"
    
    if document_aware:
        message += " (using document-aware mode)"
    else:
        message += " (not using document-aware mode)"
    
    message += " This may take some time."
    
    # Confirm extraction
    confirm = typer.confirm(message, default=True)
    if not confirm:
        console.print("[bold yellow]Operation cancelled")
        return
    
    # Create progress
    from rich.progress import Progress
    with Progress() as progress:
        # Extract events
        try:
            console.print(f"[bold green]Extracting timeline events for matter '{matter}'...")
            event_count = extract_events_from_all_documents(
                matter_id, 
                progress,
                resume=not no_resume,
                force=force,
                document_aware=document_aware,
                parallel=parallel,
                max_workers=workers
            )
            
            # Show appropriate completion message
            if event_count > 0:
                console.print(f"[bold green]Extraction complete. {event_count} events extracted.")
            else:
                if force:
                    console.print("[bold yellow]No events were extracted. Check if there are documents in this matter.")
                else:
                    console.print("[bold green]No new events to extract. Use --force to re-extract all events.")
        except Exception as e:
            console.print(f"[bold red]Error extracting timeline events: {str(e)}")
            raise typer.Exit(1)


@timeline_app.command("show")
def timeline_show_command(
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to show timeline for"
    ),
    from_date: Optional[str] = typer.Option(
        None, "--from", help="Start date (YYYY-MM-DD)"
    ),
    to_date: Optional[str] = typer.Option(
        None, "--to", help="End date (YYYY-MM-DD)"
    ),
    event_type: Optional[List[str]] = typer.Option(
        None, "--type", "-t", help="Filter by event type(s)"
    ),
    min_importance: float = typer.Option(
        0.3, "--min-importance", help="Minimum importance score (0-1)"
    ),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Minimum confidence score (0-1)"
    ),
    max_events: int = typer.Option(
        1000, "--max-events", help="Maximum number of events to show (default: 1000)"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table or text"
    ),
    include_financials: bool = typer.Option(
        True, "--financials/--no-financials", help="Include financial impact summary"
    ),
    include_contradictions: bool = typer.Option(
        True, "--contradictions/--no-contradictions", help="Include contradiction detection"
    ),
    detect_new: bool = typer.Option(
        True, "--detect-new/--no-detect-new", help="Detect new contradictions"
    ),
) -> None:
    """Show timeline for a matter."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter ID
    with get_session() as session:
        matter_obj = session.query(Matter).filter(Matter.name == matter).first()
        if not matter_obj:
            console.print(f"[bold red]Matter '{matter}' not found")
            raise typer.Exit(1)
        
        matter_id = matter_obj.id
    
    # Parse dates
    from datetime import datetime
    date_from = None
    date_to = None
    
    if from_date:
        try:
            date_from = datetime.strptime(from_date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[bold red]Invalid from date format: {from_date}. Use YYYY-MM-DD")
            raise typer.Exit(1)
    
    if to_date:
        try:
            date_to = datetime.strptime(to_date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[bold red]Invalid to date format: {to_date}. Use YYYY-MM-DD")
            raise typer.Exit(1)
    
    # Validate event types
    valid_event_types = [e.value for e in EventType]
    if event_type:
        for et in event_type:
            if et not in valid_event_types:
                console.print(f"[bold red]Invalid event type: {et}")
                console.print(f"Valid event types: {', '.join(valid_event_types)}")
                raise typer.Exit(1)
    
    # Generate timeline
    try:
        console.print(f"[bold green]Generating timeline for matter '{matter}'...")
        timeline_data = generate_claim_timeline(
            matter_id=matter_id,
            event_types=event_type,
            date_from=date_from,
            date_to=date_to,
            importance_threshold=min_importance,
            confidence_threshold=min_confidence,
            include_financial_impacts=include_financials,
            include_contradictions=include_contradictions,
            detect_new_contradictions=detect_new,
            max_events=max_events,
        )
        
        # Display timeline
        display_timeline(timeline_data, format=format)
        
    except Exception as e:
        console.print(f"[bold red]Error generating timeline: {str(e)}")
        raise typer.Exit(1)


@timeline_app.command("types")
def timeline_types_command() -> None:
    """List valid timeline event types."""
    console.print("[bold blue]Valid Timeline Event Types:")
    for event_type in EventType:
        console.print(f"  [green]{event_type.value}[/green]: {event_type.name}")


@timeline_app.command("export")
def timeline_export_command(
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to export timeline for"
    ),
    from_date: Optional[str] = typer.Option(
        None, "--from", help="Start date (YYYY-MM-DD)"
    ),
    to_date: Optional[str] = typer.Option(
        None, "--to", help="End date (YYYY-MM-DD)"
    ),
    event_type: Optional[List[str]] = typer.Option(
        None, "--type", "-t", help="Filter by event type(s)"
    ),
    min_importance: float = typer.Option(
        0.3, "--min-importance", help="Minimum importance score (0-1)"
    ),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Minimum confidence score (0-1)"
    ),
    max_events: int = typer.Option(
        1000, "--max-events", help="Maximum number of events to show (default: 1000)"
    ),
    include_financials: bool = typer.Option(
        True, "--financials/--no-financials", help="Include financial impact summary"
    ),
    include_contradictions: bool = typer.Option(
        True, "--contradictions/--no-contradictions", help="Include contradiction detection"
    ),
    detect_new: bool = typer.Option(
        True, "--detect-new/--no-detect-new", help="Detect new contradictions"
    ),
    open_pdf: bool = typer.Option(
        False, "--open", help="Open the PDF after export"
    ),
) -> None:
    """Export timeline as PDF."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter ID
    with get_session() as session:
        matter_obj = session.query(Matter).filter(Matter.name == matter).first()
        if not matter_obj:
            console.print(f"[bold red]Matter '{matter}' not found")
            raise typer.Exit(1)
        
        matter_id = matter_obj.id
    
    # Parse dates
    date_from = None
    date_to = None
    
    if from_date:
        try:
            date_from = datetime.strptime(from_date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[bold red]Invalid from date format: {from_date}. Use YYYY-MM-DD")
            raise typer.Exit(1)
    
    if to_date:
        try:
            date_to = datetime.strptime(to_date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[bold red]Invalid to date format: {to_date}. Use YYYY-MM-DD")
            raise typer.Exit(1)
    
    # Validate event types
    valid_event_types = [e.value for e in EventType]
    if event_type:
        for et in event_type:
            if et not in valid_event_types:
                console.print(f"[bold red]Invalid event type: {et}")
                console.print(f"Valid event types: {', '.join(valid_event_types)}")
                raise typer.Exit(1)
    
    # Generate timeline
    try:
        console.print(f"[bold green]Generating timeline for matter '{matter}'...")
        timeline_data = generate_claim_timeline(
            matter_id=matter_id,
            event_types=event_type,
            date_from=date_from,
            date_to=date_to,
            importance_threshold=min_importance,
            confidence_threshold=min_confidence,
            include_financial_impacts=include_financials,
            include_contradictions=include_contradictions,
            detect_new_contradictions=detect_new,
            max_events=max_events,
        )
        
        # Prepare filters dictionary for PDF metadata
        filters = {
            "event_types": event_type if event_type else [],
            "date_from": str(from_date) if from_date else None,
            "date_to": str(to_date) if to_date else None,
            "min_importance": min_importance,
            "min_confidence": min_confidence,
            "include_financials": include_financials,
            "include_contradictions": include_contradictions,
        }

        # Export as PDF
        console.print("[bold green]Exporting timeline as PDF...")
        pdf_path = export_timeline_as_pdf(timeline_data, matter, filters)
        
        if pdf_path:
            console.print(f"[bold green]Timeline exported to: [cyan]{pdf_path}[/cyan]")
            
            # Open PDF if requested
            if open_pdf:
                import subprocess
                import platform
                
                try:
                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", pdf_path], check=True)
                    elif platform.system() == "Windows":
                        subprocess.run(["start", pdf_path], shell=True, check=True)
                    else:  # Linux
                        subprocess.run(["xdg-open", pdf_path], check=True)
                except Exception as e:
                    console.print(f"[bold yellow]Could not open PDF: {str(e)}")
        else:
            console.print("[bold red]Failed to export timeline as PDF")
            
    except Exception as e:
        console.print(f"[bold red]Error exporting timeline: {str(e)}")
        raise typer.Exit(1)


@financials_app.command("summary")
def financial_summary_command(
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to show financial summary for"
    ),
    from_date: Optional[str] = typer.Option(
        None, "--from", help="Start date (YYYY-MM-DD)"
    ),
    to_date: Optional[str] = typer.Option(
        None, "--to", help="End date (YYYY-MM-DD)"
    ),
    event_type: Optional[List[str]] = typer.Option(
        None, "--type", "-t", help="Filter by event type(s)"
    ),
) -> None:
    """Show financial summary for a matter."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter ID
    with get_session() as session:
        matter_obj = session.query(Matter).filter(Matter.name == matter).first()
        if not matter_obj:
            console.print(f"[bold red]Matter '{matter}' not found")
            raise typer.Exit(1)
        
        matter_id = matter_obj.id
    
    # Parse dates
    date_from = None
    date_to = None
    
    if from_date:
        try:
            date_from = datetime.strptime(from_date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[bold red]Invalid from date format: {from_date}. Use YYYY-MM-DD")
            raise typer.Exit(1)
    
    if to_date:
        try:
            date_to = datetime.strptime(to_date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[bold red]Invalid to date format: {to_date}. Use YYYY-MM-DD")
            raise typer.Exit(1)
    
    # Validate event types
    valid_event_types = [e.value for e in EventType]
    if event_type:
        for et in event_type:
            if et not in valid_event_types:
                console.print(f"[bold red]Invalid event type: {et}")
                console.print(f"Valid event types: {', '.join(valid_event_types)}")
                raise typer.Exit(1)
    
    # Get financial summary
    try:
        console.print(f"[bold green]Getting financial summary for matter '{matter}'...")
        
        # Get financial summary
        from .database import get_financial_summary
        summary = get_financial_summary(
            matter_id=matter_id,
            event_types=event_type,
            date_from=date_from,
            date_to=date_to,
        )
        
        # Display summary
        console.rule("[bold green]Financial Impact Summary")
        console.print(f"Total Financial Impact: [bold]${summary.get('total_amount', 0):,.2f}[/bold]")
        console.print(f"Positive Impacts: [green]${summary.get('total_positive', 0):,.2f}[/green]")
        console.print(f"Negative Impacts: [red]${summary.get('total_negative', 0):,.2f}[/red]")
        console.print(f"Financial Events: [blue]{summary.get('event_count', 0)}[/blue]")
        
        # Display category breakdown
        if summary.get('event_type_totals'):
            console.print("\n[bold]Impact by Event Type:[/bold]")
            for event_type, amount in sorted(summary['event_type_totals'].items(), key=lambda x: abs(x[1]), reverse=True):
                color = "green" if amount >= 0 else "red"
                console.print(f"  {event_type}: [{color}]${amount:,.2f}[/{color}]")
                
        # Display monthly totals
        if summary.get('monthly_totals'):
            console.print("\n[bold]Monthly Financial Impact:[/bold]")
            for month, amount in sorted(summary['monthly_totals'].items()):
                year, month_num = month.split('-')
                month_name = datetime(int(year), int(month_num), 1).strftime("%B %Y")
                color = "green" if amount >= 0 else "red"
                console.print(f"  {month_name}: [{color}]${amount:,.2f}[/{color}]")
        
    except Exception as e:
        console.print(f"[bold red]Error generating financial summary: {str(e)}")
        raise typer.Exit(1)


@financials_app.command("update-totals")
def update_financial_totals_command(
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to update financial totals for"
    ),
) -> None:
    """Update running totals for financial events."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter ID
    with get_session() as session:
        matter_obj = session.query(Matter).filter(Matter.name == matter).first()
        if not matter_obj:
            console.print(f"[bold red]Matter '{matter}' not found")
            raise typer.Exit(1)
        
        matter_id = matter_obj.id
    
    # Update running totals
    try:
        console.print(f"[bold green]Updating financial running totals for matter '{matter}'...")
        
        # Update running totals
        from .database import update_running_totals
        success = update_running_totals(matter_id)
        
        if success:
            console.print("[bold green]Financial running totals updated successfully")
        else:
            console.print("[bold red]Error updating financial running totals")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]Error updating financial running totals: {str(e)}")
        raise typer.Exit(1)


@contradictions_app.command("detect")
def detect_contradictions_command(
    matter: Optional[str] = typer.Option(
        None, "--matter", "-m", help="Matter name to detect contradictions for"
    ),
    min_confidence: float = typer.Option(
        0.6, "--min-confidence", help="Minimum confidence threshold for events (0-1)"
    ),
    max_days: int = typer.Option(
        30, "--max-days", help="Maximum difference in days between events to consider them related"
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save detected contradictions to the database"
    ),
) -> None:
    """Detect contradictions between timeline events."""
    # Use current matter if not specified
    if not matter:
        matter = get_current_matter()
        if not matter:
            console.print("[bold yellow]No active matter. Use 'matter switch' or specify --matter")
            raise typer.Exit(1)
    
    # Get matter ID
    with get_session() as session:
        matter_obj = session.query(Matter).filter(Matter.name == matter).first()
        if not matter_obj:
            console.print(f"[bold red]Matter '{matter}' not found")
            raise typer.Exit(1)
        
        matter_id = matter_obj.id
    
    # Detect contradictions
    try:
        console.print(f"[bold green]Detecting contradictions for matter '{matter}'...")
        
        # Detect contradictions
        from .database import identify_contradictions, save_contradiction
        contradictions = identify_contradictions(
            matter_id=matter_id,
            min_confidence=min_confidence,
            max_date_diff_days=max_days,
        )
        
        if not contradictions:
            console.print("[green]No contradictions detected")
            return
        
        # Display contradictions
        console.print(f"[bold yellow]Detected {len(contradictions)} contradictions:")
        
        for i, contradiction in enumerate(contradictions, 1):
            event1_date = contradiction.get("event1_date", "Unknown")
            event2_date = contradiction.get("event2_date", "Unknown")
            event_type = contradiction.get("event_type", "Unknown")
            contradiction_type = contradiction.get("contradiction_type", "Unknown")
            details = contradiction.get("details", "")
            date_diff = contradiction.get("date_difference_days", 0)
            
            panel = Panel(
                f"Type: {contradiction_type}\nEvents: {event1_date} vs {event2_date} ({date_diff} days)\nDetails: {details}",
                title=f"Contradiction {i}: {event_type}",
                border_style="red"
            )
            console.print(panel)
        
        # Save contradictions if requested
        if save:
            saved_count = 0
            for contradiction in contradictions:
                if "event1_id" in contradiction and "event2_id" in contradiction:
                    success = save_contradiction(
                        event1_id=contradiction["event1_id"],
                        event2_id=contradiction["event2_id"],
                        contradiction_details=contradiction["details"],
                    )
                    if success:
                        saved_count += 1
            
            console.print(f"[bold green]{saved_count} contradictions saved to the database")
        
    except Exception as e:
        console.print(f"[bold red]Error detecting contradictions: {str(e)}")
        raise typer.Exit(1)


@contradictions_app.command("mark")
def mark_contradiction_command(
    event1_id: int = typer.Argument(..., help="ID of the first event"),
    event2_id: int = typer.Argument(..., help="ID of the second event"),
    description: str = typer.Argument(..., help="Description of the contradiction"),
) -> None:
    """Manually mark a contradiction between two events."""
    try:
        console.print(f"[bold green]Marking contradiction between events {event1_id} and {event2_id}...")
        
        # Save contradiction
        from .database import save_contradiction
        success = save_contradiction(
            event1_id=event1_id,
            event2_id=event2_id,
            contradiction_details=description,
        )
        
        if success:
            console.print("[bold green]Contradiction marked successfully")
        else:
            console.print("[bold red]Error marking contradiction")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[bold red]Error marking contradiction: {str(e)}")
        raise typer.Exit(1)


# Ensure database is initialized when module is loaded
try:
    init_database()
except Exception as e:
    console.print(f"[bold red]Warning: Error initializing database: {str(e)}[/bold red]")

# Add the commands to the app
if __name__ == "__main__":
    app()