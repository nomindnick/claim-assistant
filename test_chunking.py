#!/usr/bin/env python
"""Test script to compare different chunking methods with visualization."""

import os
import sys
from pathlib import Path
import tempfile
import webbrowser
from typing import List, Dict

import fitz  # PyMuPDF
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import typer

from claimctl.config import ensure_dirs, get_config
from claimctl.semantic_chunking import (
    create_semantic_chunks, 
    fallback_chunk_text, 
    create_hierarchical_chunks,
    create_adaptive_chunks
)

app = typer.Typer(help="Chunking visualization tool")
console = Console()


def generate_chunk_visualization(text: str, chunks: List[str], title: str) -> str:
    """Generate HTML visualization for chunk boundaries.
    
    Args:
        text: The original text
        chunks: The chunked text
        title: Title for the visualization
        
    Returns:
        HTML string with visualization
    """
    # Initialize HTML with styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chunk Visualization - {title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #2c3e50; }}
            .stats {{ margin: 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
            .chunks-container {{ margin-top: 20px; }}
            .text-content {{ white-space: pre-wrap; font-family: monospace; line-height: 1.4; }}
            .chunk {{ margin-bottom: 10px; padding: 10px; border-radius: 5px; }}
            .chunk-0 {{ background-color: #e3f2fd; }}
            .chunk-1 {{ background-color: #e8f5e9; }}
            .chunk-2 {{ background-color: #fff3e0; }}
            .chunk-3 {{ background-color: #f3e5f5; }}
            .chunk-4 {{ background-color: #e0f7fa; }}
            .chunk-5 {{ background-color: #fff8e1; }}
            .chunk-6 {{ background-color: #f1f8e9; }}
            .chunk-7 {{ background-color: #e8eaf6; }}
            .chunk-8 {{ background-color: #ffebee; }}
            .chunk-9 {{ background-color: #e0f2f1; }}
            .boundary {{ border-bottom: 2px dashed #e91e63; padding-bottom: 2px; }}
            .overlap {{ background-color: rgba(255, 152, 0, 0.2); }}
            .chunk-info {{ font-size: 0.8em; color: #333; margin-bottom: 5px; }}
            .tabs {{ display: flex; margin-bottom: 20px; }}
            .tab {{ padding: 10px 20px; cursor: pointer; background-color: #f1f1f1; margin-right: 5px; border-radius: 5px 5px 0 0; }}
            .tab.active {{ background-color: #007bff; color: white; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}
            .search-bar {{ margin: 10px 0; }}
            #search-input {{ padding: 8px; width: 300px; }}
            #search-button {{ padding: 8px 15px; background-color: #007bff; color: white; border: none; cursor: pointer; }}
            .highlight {{ background-color: yellow; }}
            .summary {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title} Chunking Visualization</h1>
            
            <div class="summary">
                <strong>Document Length:</strong> {len(text)} characters | 
                <strong>Number of Chunks:</strong> {len(chunks)} | 
                <strong>Average Chunk Size:</strong> {sum(len(c) for c in chunks) // max(1, len(chunks))} characters
            </div>
            
            <div class="search-bar">
                <input type="text" id="search-input" placeholder="Search text...">
                <button id="search-button" onclick="searchText()">Search</button>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="switchTab('visual-tab', this)">Visual Chunks</div>
                <div class="tab" onclick="switchTab('highlighted-tab', this)">Highlighted Text</div>
                <div class="tab" onclick="switchTab('raw-tab', this)">Raw Chunks</div>
            </div>
    """
    
    # Add visual chunks tab (each chunk as a separate box)
    html += """
        <div id="visual-tab" class="tab-content active">
            <div class="chunks-container">
    """
    
    for i, chunk in enumerate(chunks):
        color_index = i % 10  # Cycle through 10 different colors
        html += f"""
                <div class="chunk chunk-{color_index}">
                    <div class="chunk-info">Chunk {i+1}/{len(chunks)} - {len(chunk)} chars</div>
                    <div class="text-content">{chunk}</div>
                </div>
        """
    
    html += """
            </div>
        </div>
    """
    
    # Add highlighted text tab (original text with chunk boundaries highlighted)
    html += """
        <div id="highlighted-tab" class="tab-content">
            <div class="text-content">
    """
    
    # Track positions of each chunk in the original text
    positions = []
    for chunk in chunks:
        # Find start position of the chunk in the original text
        # This is a simplified approach and might not work perfectly for all cases
        start_pos = text.find(chunk[:min(50, len(chunk))])
        if start_pos != -1:
            positions.append((start_pos, start_pos + len(chunk)))
    
    # Sort positions by start position
    positions.sort()
    
    # Insert HTML markers for highlighting
    highlighted_text = text
    offset = 0
    for i, (start, end) in enumerate(positions):
        start += offset
        end += offset
        
        # Add start marker with class based on chunk number
        color_class = f"chunk-{i % 10}"
        start_marker = f'<span class="{color_class}">'
        highlighted_text = highlighted_text[:start] + start_marker + highlighted_text[start:]
        offset += len(start_marker)
        
        # Add end marker
        end_marker = f'</span><span class="boundary"></span>'
        highlighted_text = highlighted_text[:end + offset] + end_marker + highlighted_text[end + offset:]
        offset += len(end_marker)
    
    # Escape < and > characters except for the span tags we added
    # This is a simplified approach - in a production system you'd want to use a proper HTML parser
    html += highlighted_text.replace("&", "&amp;").replace("<span", "###SPAN_START###").replace("</span>", "###SPAN_END###").replace("<", "&lt;").replace(">", "&gt;").replace("###SPAN_START###", "<span").replace("###SPAN_END###", "</span>")
    
    html += """
            </div>
        </div>
    """
    
    # Add raw chunks tab (just text)
    html += """
        <div id="raw-tab" class="tab-content">
    """
    
    for i, chunk in enumerate(chunks):
        html += f"""
            <h3>Chunk {i+1} ({len(chunk)} chars)</h3>
            <pre>{chunk}</pre>
            <hr>
        """
    
    html += """
        </div>
    """
    
    # Add JavaScript for interactivity
    html += """
        <script>
            function switchTab(tabId, tabElement) {
                // Hide all tab contents
                const tabContents = document.getElementsByClassName('tab-content');
                for (let i = 0; i < tabContents.length; i++) {
                    tabContents[i].classList.remove('active');
                }
                
                // Remove active class from all tabs
                const tabs = document.getElementsByClassName('tab');
                for (let i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove('active');
                }
                
                // Show selected tab content and mark tab as active
                document.getElementById(tabId).classList.add('active');
                tabElement.classList.add('active');
            }
            
            function searchText() {
                const searchTerm = document.getElementById('search-input').value.toLowerCase();
                if (!searchTerm) return;
                
                // Remove existing highlights
                const highlights = document.getElementsByClassName('highlight');
                while (highlights.length > 0) {
                    const parent = highlights[0].parentNode;
                    parent.replaceChild(document.createTextNode(highlights[0].textContent), highlights[0]);
                    parent.normalize();
                }
                
                // Handle highlighted text tab
                const highlightedTab = document.getElementById('highlighted-tab');
                highlightSearchTerms(highlightedTab, searchTerm);
                
                // Handle raw chunks tab
                const rawTab = document.getElementById('raw-tab');
                highlightSearchTerms(rawTab, searchTerm);
                
                // Handle visual chunks tab
                const visualTab = document.getElementById('visual-tab');
                highlightSearchTerms(visualTab, searchTerm);
            }
            
            function highlightSearchTerms(container, searchTerm) {
                const textNodes = getTextNodes(container);
                
                textNodes.forEach(node => {
                    const text = node.nodeValue.toLowerCase();
                    if (text.includes(searchTerm)) {
                        const frag = document.createDocumentFragment();
                        let lastIdx = 0;
                        let idx;
                        
                        while ((idx = text.indexOf(searchTerm, lastIdx)) !== -1) {
                            // Add text before the match
                            if (idx > lastIdx) {
                                frag.appendChild(document.createTextNode(node.nodeValue.substring(lastIdx, idx)));
                            }
                            
                            // Add the highlighted match
                            const span = document.createElement('span');
                            span.className = 'highlight';
                            span.appendChild(document.createTextNode(node.nodeValue.substring(idx, idx + searchTerm.length)));
                            frag.appendChild(span);
                            
                            lastIdx = idx + searchTerm.length;
                        }
                        
                        // Add remaining text
                        if (lastIdx < node.nodeValue.length) {
                            frag.appendChild(document.createTextNode(node.nodeValue.substring(lastIdx)));
                        }
                        
                        // Replace original node with the fragment
                        node.parentNode.replaceChild(frag, node);
                    }
                });
            }
            
            function getTextNodes(node) {
                let textNodes = [];
                
                function getTextNodesHelper(node) {
                    if (node.nodeType === 3) {
                        textNodes.push(node);
                    } else {
                        for (let i = 0; i < node.childNodes.length; i++) {
                            getTextNodesHelper(node.childNodes[i]);
                        }
                    }
                }
                
                getTextNodesHelper(node);
                return textNodes;
            }
        </script>
    </body>
    </html>
    """
    
    return html


def create_visualization_file(text: str, chunks_dict: Dict[str, List[str]]) -> str:
    """Create an HTML file with visualizations for multiple chunking methods.
    
    Args:
        text: The original text
        chunks_dict: Dictionary mapping chunking method names to their chunks
        
    Returns:
        Path to the created HTML file
    """
    # Create a temporary HTML file
    fd, path = tempfile.mkstemp(suffix='.html', prefix='chunks_viz_')
    
    with os.fdopen(fd, 'w') as f:
        # Write HTML header with tabs for each chunking method
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chunking Methods Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                .method-tabs { display: flex; background-color: #f1f1f1; border-bottom: 1px solid #ddd; }
                .method-tab { padding: 10px 20px; cursor: pointer; }
                .method-tab.active { background-color: #007bff; color: white; }
                .method-content { display: none; padding: 0; margin: 0; height: 100vh; }
                .method-content.active { display: block; }
                iframe { width: 100%; height: 100%; border: none; }
            </style>
        </head>
        <body>
            <div class="method-tabs">
        """)
        
        # Add tabs for each chunking method
        for i, method_name in enumerate(chunks_dict.keys()):
            active_class = " active" if i == 0 else ""
            f.write(f'<div class="method-tab{active_class}" onclick="switchMethod(\'{method_name}\')">{method_name}</div>\n')
        
        f.write("</div>\n")
        
        # Add content sections for each method
        for i, (method_name, chunks) in enumerate(chunks_dict.items()):
            active_class = " active" if i == 0 else ""
            
            # Create iframe container for this method
            f.write(f'<div id="{method_name}-content" class="method-content{active_class}">\n')
            
            # Generate HTML for this chunking method
            method_html = generate_chunk_visualization(text, chunks, method_name)
            
            # Create a separate file for this method's visualization
            method_fd, method_path = tempfile.mkstemp(suffix='.html', prefix=f'chunks_{method_name}_')
            with os.fdopen(method_fd, 'w') as method_file:
                method_file.write(method_html)
            
            # Add iframe pointing to the method's HTML file
            f.write(f'<iframe src="file://{method_path}"></iframe>\n')
            f.write('</div>\n')
        
        # Add JavaScript for tab switching
        f.write("""
        <script>
            function switchMethod(methodName) {
                // Hide all method contents
                const contents = document.getElementsByClassName('method-content');
                for (let i = 0; i < contents.length; i++) {
                    contents[i].classList.remove('active');
                }
                
                // Remove active class from all tabs
                const tabs = document.getElementsByClassName('method-tab');
                for (let i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove('active');
                }
                
                // Show selected method content
                document.getElementById(methodName + '-content').classList.add('active');
                
                // Find the tab for this method and make it active
                for (let i = 0; i < tabs.length; i++) {
                    if (tabs[i].textContent === methodName) {
                        tabs[i].classList.add('active');
                        break;
                    }
                }
            }
        </script>
        </body>
        </html>
        """)
    
    return path


@app.command("compare")
def test_chunking_methods(
    pdf_path: str = typer.Argument(..., help="Path to PDF file to analyze"),
    page_num: int = typer.Option(0, "--page", "-p", help="Page number to analyze (0-based)"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Generate visual HTML comparison")
):
    """Compare different chunking methods on a specific PDF page."""
    console.print(f"[bold green]Testing chunking methods on {pdf_path}, page {page_num+1}")
    
    # Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        console.print(f"[bold red]Error opening {pdf_path}: {str(e)}")
        return
    
    # Ensure config is loaded
    config = get_config()
    chunk_size = config.chunking.CHUNK_SIZE
    chunk_overlap = config.chunking.CHUNK_OVERLAP
    
    # Check if page exists
    if page_num >= len(doc) or page_num < 0:
        console.print(f"[bold red]Error: Page {page_num+1} does not exist. PDF has {len(doc)} pages.")
        doc.close()
        return
    
    # Process requested page
    with Progress() as progress:
        task = progress.add_task("Processing page...", total=5)
        
        # Extract text
        page = doc[page_num]
        text = page.get_text("text")
        progress.update(task, advance=1, description="Extracted text")
        
        console.print(f"[bold]Page {page_num+1}: {len(text)} characters\n")
        
        # Apply different chunking methods
        chunks_dict = {}
        
        # Regular chunking
        progress.update(task, description="Regular chunking")
        regular_chunks = fallback_chunk_text(text, chunk_size, chunk_overlap)
        chunks_dict["Regular"] = regular_chunks
        progress.update(task, advance=1)
        
        # Semantic chunking
        progress.update(task, description="Semantic chunking")
        try:
            semantic_chunks = create_semantic_chunks(text, chunk_size, chunk_overlap)
            chunks_dict["Semantic"] = semantic_chunks
        except Exception as e:
            console.print(f"[bold red]Error in semantic chunking: {str(e)}")
        progress.update(task, advance=1)
        
        # Hierarchical chunking
        progress.update(task, description="Hierarchical chunking")
        try:
            hierarchical_chunks = create_hierarchical_chunks(text)
            chunks_dict["Hierarchical"] = hierarchical_chunks
        except Exception as e:
            console.print(f"[bold red]Error in hierarchical chunking: {str(e)}")
        progress.update(task, advance=1)
        
        # Adaptive chunking
        progress.update(task, description="Adaptive chunking")
        try:
            adaptive_chunks = create_adaptive_chunks(text, chunk_size, chunk_overlap, progress)
            chunks_dict["Adaptive"] = adaptive_chunks
        except Exception as e:
            console.print(f"[bold red]Error in adaptive chunking: {str(e)}")
        progress.update(task, advance=1)
        
        # Create comparison table
        comparison_table = Table(title="Chunking Methods Comparison")
        comparison_table.add_column("Method", style="bold")
        comparison_table.add_column("Chunks", justify="right")
        comparison_table.add_column("Avg Size", justify="right")
        comparison_table.add_column("Min Size", justify="right")
        comparison_table.add_column("Max Size", justify="right")
        
        for method, chunks in chunks_dict.items():
            if chunks:
                avg = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
                min_size = min(len(c) for c in chunks) if chunks else 0
                max_size = max(len(c) for c in chunks) if chunks else 0
                comparison_table.add_row(
                    method, 
                    str(len(chunks)),
                    str(avg),
                    str(min_size),
                    str(max_size)
                )
            else:
                comparison_table.add_row(method, "Error", "-", "-", "-")
        
        console.print(comparison_table)
        
        # Generate visualization if requested
        if visualize:
            progress.add_task("Generating visualization...", total=1)
            viz_path = create_visualization_file(text, chunks_dict)
            webbrowser.open(f"file://{viz_path}")
            console.print(f"[bold green]Visualization opened in browser: {viz_path}")
    
    # Close document
    doc.close()
    console.print("[bold green]Comparison complete!")


@app.command("visualize")
def visualize_single_method(
    pdf_path: str = typer.Argument(..., help="Path to PDF file to analyze"),
    method: str = typer.Option("adaptive", "--method", "-m", 
                              help="Chunking method (regular, semantic, hierarchical, adaptive)"),
    page_num: int = typer.Option(0, "--page", "-p", help="Page number to analyze (0-based)")
):
    """Visualize a single chunking method on a PDF page."""
    console.print(f"[bold green]Visualizing {method} chunking on {pdf_path}, page {page_num+1}")
    
    # Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        console.print(f"[bold red]Error opening {pdf_path}: {str(e)}")
        return
    
    # Ensure config is loaded
    config = get_config()
    chunk_size = config.chunking.CHUNK_SIZE
    chunk_overlap = config.chunking.CHUNK_OVERLAP
    
    # Check if page exists
    if page_num >= len(doc) or page_num < 0:
        console.print(f"[bold red]Error: Page {page_num+1} does not exist. PDF has {len(doc)} pages.")
        doc.close()
        return
    
    # Process requested page
    with Progress() as progress:
        task = progress.add_task("Processing page...", total=2)
        
        # Extract text
        page = doc[page_num]
        text = page.get_text("text")
        progress.update(task, advance=1, description="Extracted text")
        
        console.print(f"[bold]Page {page_num+1}: {len(text)} characters\n")
        
        # Apply requested chunking method
        chunks = []
        progress.update(task, description=f"Applying {method} chunking")
        
        try:
            if method.lower() == "regular":
                chunks = fallback_chunk_text(text, chunk_size, chunk_overlap)
            elif method.lower() == "semantic":
                chunks = create_semantic_chunks(text, chunk_size, chunk_overlap)
            elif method.lower() == "hierarchical":
                chunks = create_hierarchical_chunks(text)
            elif method.lower() == "adaptive":
                chunks = create_adaptive_chunks(text, chunk_size, chunk_overlap, progress)
            else:
                console.print(f"[bold red]Unknown chunking method: {method}")
                doc.close()
                return
        except Exception as e:
            console.print(f"[bold red]Error in {method} chunking: {str(e)}")
            doc.close()
            return
        
        progress.update(task, advance=1)
        
        # Generate visualization
        method_name = method.capitalize()
        html = generate_chunk_visualization(text, chunks, method_name)
        
        # Write HTML to file and open in browser
        fd, path = tempfile.mkstemp(suffix='.html', prefix=f'chunks_{method}_')
        with os.fdopen(fd, 'w') as f:
            f.write(html)
        
        webbrowser.open(f"file://{path}")
        console.print(f"[bold green]Visualization opened in browser: {path}")
    
    # Close document
    doc.close()


if __name__ == "__main__":
    # Configure Typer for pretty help text
    app()