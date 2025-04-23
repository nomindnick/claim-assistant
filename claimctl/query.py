"""Query module for claim-assistant."""

import json
import os
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from .config import get_config
from .search import search_documents as search_docs
from .utils import console

# Question answering prompt
QA_PROMPT = """
You are a construction claim assistant. Your task is to answer questions about construction claims using only the provided document chunks. 

Document chunks:
{chunks}

Answer the following question in detail, citing specific documents, dates, and page numbers when relevant. Be clear about what information comes from which source.
If the provided chunks don't contain enough information to answer the question confidently, say so clearly rather than speculating.

Question: {question}
"""


def parse_date(date_str: str) -> Optional[date]:
    """Try to parse a date string in various formats."""
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    return None


def search_documents(
    query: str,
    top_k: Optional[int] = None,
    doc_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    project_name: Optional[str] = None,
    parties: Optional[str] = None,
    search_type: str = "hybrid",
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Search for documents relevant to the query."""
    config = get_config()
    if not top_k:
        top_k = config.retrieval.TOP_K

    # Prepare metadata filters
    metadata_filters = {}
    if doc_type:
        metadata_filters["document_type"] = doc_type
    if project_name:
        metadata_filters["project_name"] = project_name
    if parties:
        metadata_filters["parties_involved"] = parties

    # Parse date filters
    if date_from:
        parsed_date = parse_date(date_from)
        if parsed_date:
            metadata_filters["date_from"] = parsed_date

    if date_to:
        parsed_date = parse_date(date_to)
        if parsed_date:
            metadata_filters["date_to"] = parsed_date

    # Execute search
    chunks, scores = search_docs(
        query=query,
        top_k=top_k,
        metadata_filters=metadata_filters if metadata_filters else None,
        search_type=search_type,
    )

    return chunks, scores


def answer_question(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Generate an answer to the question using the provided chunks."""
    config = get_config()

    # Format chunks for prompt
    formatted_chunks = ""
    for i, chunk in enumerate(chunks):
        formatted_chunks += (
            f"DOCUMENT {i+1}: {chunk['file_name']} (Page {chunk['page_num']})\n"
        )
        formatted_chunks += f"Document Type: {chunk['chunk_type']}\n"
        if chunk.get("doc_id"):
            formatted_chunks += f"Document ID: {chunk['doc_id']}\n"
        if chunk.get("doc_date"):
            formatted_chunks += f"Date: {chunk['doc_date']}\n"
        if chunk.get("project_name"):
            formatted_chunks += f"Project: {chunk['project_name']}\n"
        if chunk.get("parties_involved"):
            formatted_chunks += f"Parties: {chunk['parties_involved']}\n"
        formatted_chunks += f"Text: {chunk['text'][:800]}...\n\n"

    # Create prompt
    prompt = QA_PROMPT.format(
        chunks=formatted_chunks,
        question=question,
    )

    try:
        # Query GPT-4o-mini
        client = OpenAI(api_key=config.openai.API_KEY)
        response = client.chat.completions.create(
            model=config.openai.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a construction claim assistant skilled at answering questions based on provided document excerpts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content
    except Exception as e:
        console.log(f"[bold red]Error generating answer: {str(e)}")
        return f"Sorry, I couldn't generate an answer due to an error: {str(e)}. Please check the source documents manually."


def display_results(
    question: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    scores: List[float],
    json_output: bool = False,
    markdown_output: bool = False,
) -> None:
    """Display the search results and answer."""
    if json_output:
        # Format results as JSON
        results = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "file_name": chunk["file_name"],
                    "page_num": chunk["page_num"],
                    "chunk_type": chunk["chunk_type"],
                    "score": score,
                    "project_name": chunk.get("project_name"),
                    "doc_date": chunk.get("doc_date"),
                    "parties_involved": chunk.get("parties_involved"),
                    "text": chunk["text"][:200] + "...",  # Truncate for readability
                    "image_path": chunk.get("image_path"),
                }
                for chunk, score in zip(chunks, scores)
            ],
        }
        console.print(json.dumps(results, indent=2))
        return

    if markdown_output:
        # Format results as Markdown
        md = f"# Question\n\n{question}\n\n# Answer\n\n{answer}\n\n# Sources\n\n"
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            md += (
                f"## Source {i+1}: {chunk['file_name']} (Page {chunk['page_num']})\n\n"
            )
            md += f"- **Type:** {chunk['chunk_type']}\n"
            md += f"- **Relevance:** {score:.2f}\n"
            if chunk.get("project_name"):
                md += f"- **Project:** {chunk['project_name']}\n"
            if chunk.get("doc_date"):
                md += f"- **Date:** {chunk['doc_date']}\n"
            if chunk.get("parties_involved"):
                md += f"- **Parties:** {chunk['parties_involved']}\n"
            md += f"- **Preview:** {chunk['text'][:200]}...\n\n"
        console.print(md)
        return

    # Regular rich console output
    console.rule("[bold blue]Question")
    console.print(question)

    console.rule("[bold green]Answer")
    console.print(Markdown(answer))

    console.rule("[bold yellow]Sources")
    table = Table(title="Matching Documents")
    table.add_column("#", style="dim")
    table.add_column("File")
    table.add_column("Page")
    table.add_column("Type")
    table.add_column("Project")
    table.add_column("Date")
    table.add_column("Relevance")

    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        table.add_row(
            str(i + 1),
            chunk["file_name"],
            str(chunk["page_num"]),
            chunk["chunk_type"],
            chunk.get("project_name", ""),
            str(chunk.get("doc_date", "")),
            f"{score:.2f}",
        )

    console.print(table)
    console.print("\nCommands: (o)pen PDF, (e)xport image to exhibits, (q)uit")


def handle_user_commands(chunks: List[Dict[str, Any]]) -> None:
    """Handle user commands for interacting with search results."""
    while True:
        choice = Prompt.ask("Enter command").lower()

        if choice == "q":
            break

        elif choice.startswith("o"):
            # Extract source number if provided (e.g., "o 2" opens source #2)
            parts = choice.split()
            src_num = 1  # Default to first source
            if len(parts) > 1 and parts[1].isdigit():
                src_num = int(parts[1])

            if 1 <= src_num <= len(chunks):
                chunk = chunks[src_num - 1]
                # Open PDF to the specific page
                try:
                    file_path = chunk["file_path"]
                    page_num = chunk["page_num"]

                    if os.name == "posix":  # Linux/Mac
                        cmd = ["xdg-open" if os.name == "posix" else "start", file_path]
                        subprocess.Popen(cmd)
                        console.print(f"Opening {file_path} page {page_num}")
                    elif os.name == "nt":  # Windows
                        cmd = ["start", "", file_path]
                        subprocess.Popen(cmd, shell=True)
                        console.print(f"Opening {file_path} page {page_num}")
                    else:
                        console.print("Unsupported operating system")
                except Exception as e:
                    console.print(f"[bold red]Error opening PDF: {str(e)}")
            else:
                console.print(f"[bold red]Invalid source number: {src_num}")

        elif choice.startswith("e"):
            # Extract source number if provided
            parts = choice.split()
            src_num = 1  # Default to first source
            if len(parts) > 1 and parts[1].isdigit():
                src_num = int(parts[1])

            if 1 <= src_num <= len(chunks):
                chunk = chunks[src_num - 1]
                # Export image to exhibits directory
                try:
                    image_path = chunk.get("image_path")
                    if not image_path:
                        console.print("[bold yellow]No image available for this chunk")
                        continue

                    # Create exhibits directory if it doesn't exist
                    exhibits_dir = Path("./exhibits")
                    exhibits_dir.mkdir(exist_ok=True)

                    # Copy image to exhibits directory
                    import shutil

                    target_path = exhibits_dir / Path(image_path).name
                    shutil.copy2(image_path, target_path)

                    console.print(f"[bold green]Exported to {target_path}")
                except Exception as e:
                    console.print(f"[bold red]Error exporting image: {str(e)}")
            else:
                console.print(f"[bold red]Invalid source number: {src_num}")

        else:
            console.print("[bold red]Unknown command. Use (o)pen, (e)xport, or (q)uit.")


def query_documents(
    question: str,
    top_k: Optional[int] = None,
    json_output: bool = False,
    markdown_output: bool = False,
    doc_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    project_name: Optional[str] = None,
    parties: Optional[str] = None,
    search_type: str = "hybrid",
) -> None:
    """Query documents based on a natural language question."""
    # Search for relevant documents
    chunks, scores = search_documents(
        question,
        top_k,
        doc_type,
        date_from,
        date_to,
        project_name,
        parties,
        search_type,
    )

    if not chunks:
        console.print(
            "[bold red]No relevant documents found. Have you ingested any PDFs?"
        )
        return

    # Generate answer
    answer = answer_question(question, chunks)

    # Display results
    display_results(question, answer, chunks, scores, json_output, markdown_output)

    # Handle user commands (unless output format is specified)
    if not json_output and not markdown_output:
        handle_user_commands(chunks)
