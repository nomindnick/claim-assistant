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

Answer the following question in detail. Follow these requirements:

1. Use only information from the provided documents
2. Include specific document references with each claim (e.g., "According to [Doc 3, p.5]...")
3. Include document metadata like dates, project names, and parties when relevant
4. Format citations consistently as [Doc X, p.Y] directly in-line with the text
5. For each major claim in your answer, indicate confidence level as:
   - [HIGH CONFIDENCE]: When multiple documents support the claim or evidence is very clear
   - [MEDIUM CONFIDENCE]: When evidence exists but is limited to a single source
   - [LOW CONFIDENCE]: When information is implied but not explicitly stated
6. If the provided chunks don't contain enough information to answer the question, say so clearly

After your answer, add a "Sources" section that lists the documents you referenced, with the most relevant ones first.

Question: {question}

{follow_up_context}
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


def answer_question(
    question: str, 
    chunks: List[Dict[str, Any]], 
    follow_up_context: Optional[str] = None
) -> str:
    """Generate an answer to the question using the provided chunks."""
    config = get_config()
    context_size = config.retrieval.CONTEXT_SIZE
    
    # Format chunks for prompt with enhanced metadata
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
        
        # Use full chunk text up to context_size limit - improved from 800 chars
        formatted_chunks += f"Text: {chunk['text'][:context_size]}"
        
        # Add text length indicator if truncated
        if len(chunk['text']) > context_size:
            formatted_chunks += f"... [truncated, {len(chunk['text']) - context_size} more characters]"
        
        formatted_chunks += "\n\n"

    # Create prompt
    prompt = QA_PROMPT.format(
        chunks=formatted_chunks,
        question=question,
        follow_up_context=follow_up_context or "",
    )

    try:
        # Query OpenAI with expanded system prompt
        client = OpenAI(api_key=config.openai.API_KEY)
        response = client.chat.completions.create(
            model=config.openai.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a construction claim assistant skilled at answering questions based on provided document excerpts.
                    
Key characteristics of your responses:
1. Thorough analysis of provided documents
2. Clear, specific citations for every claim
3. Appropriate confidence indicators for claims
4. Honest indication of information gaps
5. Effective use of metadata (dates, parties, etc.) to contextualize information
                    """,
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
    console.print("\nCommands: (f)ollow-up, (c)ompare, (m)ore like this, (o)pen PDF, (e)xport image, (q)uit")


def handle_user_commands(
    question: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    doc_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    project_name: Optional[str] = None,
    parties: Optional[str] = None,
    search_type: str = "hybrid",
) -> None:
    """Handle user commands for interacting with search results."""
    previous_answer_context = None
    while True:
        console.print("\nCommands: (f)ollow-up, (c)ompare, (m)ore like this, (o)pen PDF, (e)xport image, (q)uit")
        choice = Prompt.ask("Enter command").lower()

        if choice == "q":
            break

        elif choice.startswith("f"):
            # Follow-up question
            follow_up = Prompt.ask("Enter your follow-up question")
            
            # Build context to maintain conversation flow
            if previous_answer_context:
                follow_up_context = f"Previous question: {question}\n{previous_answer_context}\n\nFollow-up question: {follow_up}"
            else:
                follow_up_context = f"This is a follow-up to: {question}"
            
            # Search with the follow-up question but maintain conversation context
            try:
                new_chunks, new_scores = search_documents(
                    follow_up,
                    top_k,
                    doc_type,
                    date_from,
                    date_to,
                    project_name,
                    parties,
                    search_type,
                )
                
                if not new_chunks:
                    console.print("[bold red]No relevant documents found for your follow-up.")
                    continue
                
                # Generate answer with follow-up context
                answer = answer_question(follow_up, new_chunks, follow_up_context)
                
                # Store for next follow-up
                previous_answer_context = f"User asked: {follow_up}\nAssistant answered: {answer}"
                
                # Display results
                display_results(follow_up, answer, new_chunks, new_scores)
                
                # Update chunks for other commands
                chunks = new_chunks
                question = follow_up
            except Exception as e:
                console.print(f"[bold red]Error processing follow-up: {str(e)}")

        elif choice.startswith("c"):
            # Compare documents
            console.print("Select two sources to compare:")
            src_num1 = Prompt.ask("First source #", default="1")
            src_num2 = Prompt.ask("Second source #", default="2")
            
            try:
                idx1 = int(src_num1) - 1
                idx2 = int(src_num2) - 1
                
                if 0 <= idx1 < len(chunks) and 0 <= idx2 < len(chunks):
                    chunk1 = chunks[idx1]
                    chunk2 = chunks[idx2]
                    
                    # Create comparison prompt
                    comparison_prompt = f"""
                    Please compare these two documents about the same topic:
                    
                    DOCUMENT A: {chunk1['file_name']} (Page {chunk1['page_num']})
                    Date: {chunk1.get('doc_date', 'Unknown')}
                    Text: {chunk1['text'][:1000]}...
                    
                    DOCUMENT B: {chunk2['file_name']} (Page {chunk2['page_num']})
                    Date: {chunk2.get('doc_date', 'Unknown')}
                    Text: {chunk2['text'][:1000]}...
                    
                    Analyze the following:
                    1. Key similarities between these documents
                    2. Important differences or contradictions
                    3. How the information might complement each other
                    4. Which document appears more authoritative if they contradict
                    """
                    
                    # Get config
                    config = get_config()
                    
                    # Query GPT-4o-mini
                    client = OpenAI(api_key=config.openai.API_KEY)
                    response = client.chat.completions.create(
                        model=config.openai.MODEL,
                        messages=[
                            {"role": "system", "content": "You are an assistant that compares construction documents."},
                            {"role": "user", "content": comparison_prompt},
                        ],
                        temperature=0.0,
                    )
                    
                    comparison = response.choices[0].message.content
                    console.rule("[bold blue]Document Comparison")
                    console.print(Markdown(comparison))
                else:
                    console.print("[bold red]Invalid source number(s).")
            except ValueError:
                console.print("[bold red]Please enter valid source numbers.")
            except Exception as e:
                console.print(f"[bold red]Error comparing documents: {str(e)}")

        elif choice.startswith("m"):
            # More like this - find similar documents to a specific source
            src_num = Prompt.ask("Enter source # to find similar documents", default="1")
            
            try:
                idx = int(src_num) - 1
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    
                    # Use the chunk text as a query to find similar documents
                    console.print(f"[bold green]Finding documents similar to {chunk['file_name']} (Page {chunk['page_num']})...")
                    
                    similar_chunks, scores = search_documents(
                        chunk['text'][:500],  # Use first 500 chars as query
                        top_k,
                        doc_type,
                        date_from,
                        date_to,
                        project_name,
                        parties,
                        search_type,
                    )
                    
                    # Filter out the original document
                    similar_chunks = [c for c in similar_chunks if not (
                        c['file_path'] == chunk['file_path'] and 
                        c['page_num'] == chunk['page_num']
                    )]
                    
                    if not similar_chunks:
                        console.print("[bold yellow]No other similar documents found.")
                        continue
                    
                    # Display similar documents
                    console.rule("[bold blue]Similar Documents")
                    table = Table(title=f"Documents Similar to {chunk['file_name']} (Page {chunk['page_num']})")
                    table.add_column("#", style="dim")
                    table.add_column("File")
                    table.add_column("Page")
                    table.add_column("Type")
                    table.add_column("Project")
                    table.add_column("Relevance")
                    
                    for i, (similar, score) in enumerate(zip(similar_chunks, scores)):
                        table.add_row(
                            str(i + 1),
                            similar["file_name"],
                            str(similar["page_num"]),
                            similar["chunk_type"],
                            similar.get("project_name", ""),
                            f"{score:.2f}",
                        )
                    
                    console.print(table)
                    
                    # Ask if user wants to update their current context
                    update = Prompt.ask(
                        "Update current context with these documents? (y/n)", 
                        choices=["y", "n"], 
                        default="n"
                    )
                    
                    if update.lower() == "y":
                        chunks = similar_chunks
                        console.print("[bold green]Context updated with similar documents.")
                else:
                    console.print("[bold red]Invalid source number.")
            except ValueError:
                console.print("[bold red]Please enter a valid source number.")
            except Exception as e:
                console.print(f"[bold red]Error finding similar documents: {str(e)}")

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
            console.print("[bold red]Unknown command. Use (f)ollow-up, (c)ompare, (m)ore like this, (o)pen, (e)xport, or (q)uit.")


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

    # Generate answer (with no follow-up context for the initial question)
    answer = answer_question(question, chunks)

    # Display results
    display_results(question, answer, chunks, scores, json_output, markdown_output)

    # Handle user commands (unless output format is specified)
    if not json_output and not markdown_output:
        handle_user_commands(
            question,
            chunks,
            top_k,
            doc_type,
            date_from,
            date_to,
            project_name,
            parties,
            search_type,
        )
