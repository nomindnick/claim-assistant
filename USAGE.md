# Construction Claim Assistant Usage Guide

This document provides instructions for using the Construction Claim Assistant CLI application.

## Getting Started

### Setup and Installation

1. **Activate the virtual environment**:
   ```bash
   cd ~/Projects/claim-assistant
   source venv/bin/activate
   ```

2. **Verify your configuration**:
   ```bash
   python -m claimctl.cli config show
   ```
   
   Your OpenAI API key should be set in the configuration.

### Using the Interactive Shell

The interactive shell provides a more user-friendly way to interact with the application:

1. **Launch the shell**:
   ```bash
   ./run.sh
   ```
   
   This will start the interactive shell with command history, auto-completion, and visual context.

2. **Command Completion**:
   - Press Tab to complete commands and see available options
   - Use arrow keys to navigate command history
   - Command parameters and subcommands are auto-completed

3. **Simplified Syntax**:
   - In the shell, you can use simpler syntax (e.g., `ask "What caused the delay?"` instead of `python -m claimctl.cli ask "What caused the delay?"`)
   - The shell automatically handles formatting commands correctly

4. **Contextual Information**:
   - The shell prompt shows the currently active matter
   - Help is available by typing `help`
   - Exit with `exit` or `quit`

## Key Features

The application provides several powerful capabilities for construction claim analysis:

1. **Document Ingestion and Analysis**: Efficiently process, classify, and extract information from PDF documents
2. **Natural Language Querying**: Ask questions in plain English about your construction project
3. **Timeline Generation**: Extract and visualize chronological events from your documents
4. **Financial Impact Analysis**: Track monetary changes throughout your project timeline
5. **Contradiction Detection**: Identify conflicting information between documents

## Working with Documents

### Ingesting PDFs

Before you can query your documents, you need to ingest them:

```bash
# Ingest a specific PDF
python -m claimctl.cli ingest /path/to/your/document.pdf

# Ingest multiple PDFs
python -m claimctl.cli ingest /path/to/file1.pdf /path/to/file2.pdf

# Ingest all PDFs in a directory
python -m claimctl.cli ingest /path/to/directory/*.pdf

# Process a directory of PDFs (including subdirectories)
python -m claimctl.cli ingest /path/to/directory --recursive

# Process in batches with progress tracking
python -m claimctl.cli ingest /path/to/directory --batch-size 10

# Resume processing from a previous interrupted run
python -m claimctl.cli ingest /path/to/directory --resume

# Control chunking methods
python -m claimctl.cli ingest /path/to/directory --adaptive-chunking
python -m claimctl.cli ingest /path/to/directory --no-semantic-chunking
python -m claimctl.cli ingest /path/to/directory --no-hierarchical-chunking

# Control ingestion logging
python -m claimctl.cli ingest /path/to/directory --logging      # Enable detailed logging (default)
python -m claimctl.cli ingest /path/to/directory --no-logging   # Disable detailed logging

# Control automatic timeline extraction
python -m claimctl.cli ingest /path/to/directory --timeline-extract     # Enable timeline extraction (default)
python -m claimctl.cli ingest /path/to/directory --no-timeline-extract  # Disable timeline extraction
```

During ingestion, the system:
- Extracts text from each page
- Runs OCR if needed for scanned content
- Saves page images for reference
- Classifies document types (Email, ChangeOrder, Invoice, etc.)
- Extracts enhanced metadata (amounts, time periods, section references, etc.)
- Analyzes document structure for optimal chunking
- Creates semantic chunks respecting document boundaries
- Generates embeddings for semantic search
- Automatically extracts timeline events from document chunks (if enabled)
- Stores everything in the database

### Asking Questions

Once your documents are ingested, you can ask natural language questions:

```bash
python -m claimctl.cli ask "Where is Change Order 12 justified?"

# Specify the number of documents to use (default is 20)
python -m claimctl.cli ask "Where is Change Order 12 justified?" --top-k 30
# or using the shorter flag
python -m claimctl.cli ask "Where is Change Order 12 justified?" -k 10
```

The system will:
1. Find relevant documents based on semantic similarity (by default, the top 25 documents)
2. Rerank results using a cross-encoder model for more accurate relevance scoring
3. Extract and analyze metadata from documents (amounts, time periods, section references, etc.)
4. Generate a comprehensive answer using GPT-4o-mini
5. Display source documents with their relevance scores and extracted metadata

### Interactive Commands

After seeing the results, you can interact with the documents:

- Type `f` to ask a follow-up question with context from previous questions
- Type `c` to compare two documents from your results
- Type `m` to find more documents similar to a specific result
- Type `s` to sort/filter your results by different criteria
- Type `v` to view an image of the document page
- Type `o` to open the most relevant PDF
- Type `o 2` to open the second most relevant PDF
- Type `e` to export the most relevant page as an image to the exhibits folder
- Type `e 3` to export the third most relevant page
- Type `p` to export the full response as a PDF with all referenced documents
- Type `q` to quit

### Output Formats

You can get results in different formats:

```bash
# Standard interactive output
python -m claimctl.cli ask "When was the project delayed?"

# JSON output (useful for programmatic use)
python -m claimctl.cli ask "What was the project completion date?" --json

# Markdown output (good for reports)
python -m claimctl.cli ask "What caused the delay?" --md
```

## Advanced Usage

### Ingestion Logging and Analysis

The system provides detailed logging of document ingestion to help track processing metrics, classification decisions, and potential issues:

```bash
# View recent ingestion logs
python -m claimctl.cli logs list

# Limit the number of logs shown
python -m claimctl.cli logs list --limit 10

# View logs for a specific matter
python -m claimctl.cli logs list --matter "Highway Project"

# Show detailed summary of the most recent log
python -m claimctl.cli logs show

# Show summary of a specific log file
python -m claimctl.cli logs show ingestion_20250503_123456.jsonl

# Show log for a specific matter
python -m claimctl.cli logs show --matter "Highway Project"
```

The ingestion logs capture:

- **Document Processing Metrics**: Pages processed, chunks created, processing times
- **Classification Decisions**: Document type classifications with confidence scores
- **Metadata Extraction**: Success rates for various metadata fields
- **Chunking Methods**: Which chunking strategies were applied to each document
- **Errors and Warnings**: Any issues encountered during processing

These logs help identify:
- Documents that caused processing problems
- Classification patterns and distribution
- Extraction performance for different document types
- Processing bottlenecks and optimization opportunities

The logs are stored as JSONL files in each matter's `logs` directory, allowing for easy parsing and analysis with external tools if needed.

### Advanced Document Chunking

The system uses several sophisticated methods for dividing documents into chunks:

```bash
# Use all chunking methods (default)
python -m claimctl.cli ingest documents/*.pdf

# Use adaptive chunking that automatically detects document structure
python -m claimctl.cli ingest documents/*.pdf --adaptive-chunking

# Use only semantic chunking (no hierarchical or adaptive)
python -m claimctl.cli ingest documents/*.pdf --no-hierarchical-chunking --no-adaptive-chunking

# Disable semantic chunking and use traditional character-based chunking
python -m claimctl.cli ingest documents/*.pdf --no-semantic-chunking
```

#### Chunking Methods Explained

1. **Regular Chunking**: Traditional method that splits text by character count with attention to natural separators like paragraphs.

2. **Semantic Chunking**: Uses embeddings to identify natural semantic boundaries in text, keeping related content together even if it exceeds standard character limits.

3. **Hierarchical Chunking**: Creates a hierarchical representation of structured documents (like contracts) to preserve section relationships.

4. **Adaptive Chunking**: Automatically analyzes document structure and selects the optimal chunking method based on content type.

For large documents (>500K characters), the system automatically uses memory-optimized processing that:
- Segments the document into manageable parts
- Processes each segment using the appropriate chunking method
- Removes duplicate chunks at segment boundaries
- Provides progress tracking throughout the process

### Chunk Visualization Tool

A specialized tool is included to help visualize and compare different chunking methods:

```bash
# Compare all chunking methods on a specific PDF page
python test_chunking.py compare path/to/document.pdf --page 0

# Visualize a single chunking method
python test_chunking.py visualize path/to/document.pdf --method adaptive
```

The visualization provides:
- Color-coded visual representation of chunks
- Highlighted text showing exact chunk boundaries
- Statistical comparison of chunk sizes and distribution
- Search functionality to find text across chunks

This tool is particularly useful for understanding how different chunking strategies affect document representation and retrieval.

### Cross-Encoder Reranking

The system uses a cross-encoder model to improve search result relevance:

```bash
# Enable reranking (default)
python -m claimctl.cli ask "What caused the delay?" 

# Disable reranking for faster results (less accurate)
python -m claimctl.cli ask "What caused the delay?" --no-rerank
```

Cross-encoder models provide more accurate relevance scoring by processing query-document pairs through a single model, rather than encoding them separately. This approach:

- Improves ranking precision for complex queries
- Provides better handling of queries with multiple concepts or technical terms
- Can distinguish between documents that merely mention relevant terms and those that directly address the query
- Is particularly effective at identifying relevant clauses in contracts and legal documents

The system uses the 'cross-encoder/ms-marco-MiniLM-L-6-v2' model by default, which offers a good balance between performance and accuracy.

### Filtering and Refining Queries

You can refine your searches with various filters:

```bash
# Filter by document type
python -m claimctl.cli ask "What are the material costs?" --type Invoice

# Filter by project name
python -m claimctl.cli ask "What caused the delay?" --project "Highway Bridge Project"

# Filter by date range
python -m claimctl.cli ask "What were the issues?" --from "2024-01-01" --to "2024-04-01"

# Filter by parties involved
python -m claimctl.cli ask "What were the contract terms?" --parties "ABC Construction"

# Filter by monetary amount range
python -m claimctl.cli ask "What changes were approved?" --amount-min 10000 --amount-max 50000

# Filter by contract section reference
python -m claimctl.cli ask "What does the contract say about delays?" --section "3.2.1"

# Only show public agency documents
python -m claimctl.cli ask "What was the board's decision?" --public-agency

# Change search type (default is hybrid)
python -m claimctl.cli ask "What is the total cost?" --search vector  # Options: hybrid, vector, keyword
```

### Document Visualization and Result Management

The system provides several enhancements for working with results:

* **Document Thumbnails**: Thumbnails are automatically generated during ingestion and can be viewed with the `v` command
* **Text Highlighting**: Search terms are automatically highlighted in the results
* **Result Sorting**: Type `s` to sort results by relevance, date, or document type
* **Result Filtering**: Filter results to show only specific document types
* **Document Comparison**: Type `c` to compare two documents and analyze their similarities and differences
* **Follow-up Questions**: Type `f` to ask follow-up questions while maintaining conversation context
* **PDF Export**: Type `p` to export the full response with all referenced documents as a PDF file for later reference

### Matter Management

The system supports working with multiple legal matters simultaneously:

```bash
# Create a new matter
python -m claimctl.cli matter create "Smith Construction Claim" --desc "Highway project delay claim"

# List all matters
python -m claimctl.cli matter list

# Switch to a different matter
python -m claimctl.cli matter switch "Smith Construction Claim"

# See matter information
python -m claimctl.cli matter info

# Delete a matter (and optionally its directories)
python -m claimctl.cli matter delete "Smith Construction Claim"
```

When working with matters:

- Each matter has its own document storage and search index
- You must specify or switch to a matter before ingesting documents
- Documents ingested into one matter are not visible to other matters
- The current active matter is used by default for all operations

```bash
# Ingest documents into a specific matter
python -m claimctl.cli ingest /path/to/pdfs --matter "Smith Construction Claim" 

# Query a specific matter
python -m claimctl.cli ask "What caused the delay?" --matter "Jones Project"
```

# Configuration Options

You can edit `~/.claimctl.ini` to change:
- Data storage locations (DATA_DIR, INDEX_DIR)
- OpenAI model selection (MODEL, EMBED_MODEL)
- Retrieval parameters:
  - TOP_K: Number of documents to retrieve (default is 20, which pulls ~200 initial candidates)
  - SCORE_THRESHOLD: Minimum similarity score for documents
  - CONTEXT_SIZE: Character limit per chunk for context window
  - ANSWER_CONFIDENCE: Whether to include confidence indicators
  - RERANK_ENABLED: Whether to use cross-encoder reranking (default is True)
- Chunking parameters:
  - CHUNK_SIZE: Maximum chunk size in characters
  - CHUNK_OVERLAP: Overlap between chunks in characters
  - SEMANTIC_CHUNKING: Enable semantic chunking (default is True)
  - HIERARCHICAL_CHUNKING: Enable hierarchical chunking for structured documents (default is True)
  - ADAPTIVE_CHUNKING: Enable automatic structure detection (default is True)
  - LARGE_DOC_THRESHOLD: Character threshold for large document optimization (default is 500000)
  - SIMILARITY_THRESHOLD: Threshold for detecting duplicate chunks (default is 0.8)
- BM25 search parameters (K1, B, WEIGHT)
- Project settings (DEFAULT_PROJECT)
- Matter settings (MATTER_DIR, CURRENT_MATTER)
- Timeline settings:
  - AUTO_EXTRACT: Enable automatic timeline extraction during ingestion (default is True)
  - EXTRACT_CONFIDENCE_THRESHOLD: Minimum confidence score for extracted events (default is 0.5)
  - EXTRACT_IMPORTANCE_THRESHOLD: Minimum importance score for extracted events (default is 0.3)
  - EXTRACTION_BATCH_SIZE: Number of chunks to process in a batch (default is 10)

### Environment Variables

You can also override config values with environment variables:

```bash
CLAIMCTL_OPENAI_API_KEY=your-key python -m claimctl.cli ask "..."
```

## Troubleshooting

If you encounter issues:

1. **API Connection Problems**:
   - Verify your API key is correct in `~/.claimctl.ini`
   - Check your internet connection
   - Ensure you have credit on your OpenAI account

2. **No Results Found**:
   - Make sure you've ingested documents that contain relevant information
   - Try rephrasing your question
   - Use broader terms if you're getting too few results

3. **Tesseract OCR Issues**:
   - Ensure Tesseract is installed: `tesseract --version`
   - If getting OCR errors, you may need to install Tesseract:
     ```bash
     sudo apt-get install tesseract-ocr  # Ubuntu/Debian
     brew install tesseract  # macOS
     ```

### Clearing Data and Starting Fresh

You can clear previously ingested data and start fresh using the `clear` command:

```bash
# Clear everything (database, embeddings, images, cache files, etc.)
python -m claimctl.cli clear --all

# Clear only specific components
python -m claimctl.cli clear --database --embeddings

# Clear different types of files
python -m claimctl.cli clear --images      # Clear full-size images and thumbnails
python -m claimctl.cli clear --cache       # Clear temporary cache files
python -m claimctl.cli clear --exhibits    # Clear exported exhibits
python -m claimctl.cli clear --exports     # Clear exported response PDFs
python -m claimctl.cli clear --resume-log  # Clear the ingestion resume log
```

The clear command will ask for confirmation before deleting any data and will provide a report of what was cleared.

### Working with Timelines

The system can extract timeline events from your documents and create comprehensive claim chronologies. Timeline events are automatically extracted during document ingestion (configurable with `--timeline-extract/--no-timeline-extract`), but you can also run the extraction process manually:

```bash
# Extract timeline events from all documents in the current matter (if not already extracted during ingestion)
python -m claimctl.cli timeline extract

# Show the timeline for the current matter
python -m claimctl.cli timeline show

# Show timeline with specific parameters
python -m claimctl.cli timeline show --from "2024-01-01" --to "2024-05-31" --type change_order --min-importance 0.4

# Export timeline as PDF
python -m claimctl.cli timeline export --open  # Add --open to automatically open the PDF

# Show financial summary
python -m claimctl.cli timeline financials summary

# Show available event types
python -m claimctl.cli timeline types

# Detect contradictions between timeline events
python -m claimctl.cli timeline contradictions detect
```

#### Timeline Event Types

The system supports various event types including:
- `project_start`: Project commencement events
- `project_completion`: Project completion milestones
- `change_order`: Contract modifications and change orders
- `delay`: Schedule impacts and delay events
- `payment`: Payment transactions and invoices
- `notice`: Formal notices between parties
- `claim`: Claim submissions and responses
- `dispute`: Disputed matters and resolutions
- `agreement`: Contract and agreement events
- `correspondence`: Important correspondence between parties
- `meeting`: Meeting minutes and discussions
- `request_for_information`: RFIs and responses
- `other`: Miscellaneous events

#### Timeline Display Formats

You can display timelines in different formats:

```bash
# Show timeline in table format (default)
python -m claimctl.cli timeline show --format table

# Show timeline in text format with chronological grouping
python -m claimctl.cli timeline show --format text
```

#### Financial Impact Analysis

The system tracks financial impacts across various event types:

```bash
# Show financial summary for all events
python -m claimctl.cli timeline financials summary

# Show financial summary for specific event types
python -m claimctl.cli timeline financials summary --type change_order --type claim

# Show financial summary for a date range
python -m claimctl.cli timeline financials summary --from "2024-01-01" --to "2024-05-31"

# Update running financial totals (recalculate chronological running sums)
python -m claimctl.cli timeline financials update-totals
```

#### Contradiction Detection

The system can identify contradictions between events in your timeline:

```bash
# Detect contradictions with default settings
python -m claimctl.cli timeline contradictions detect

# Detect contradictions with specific parameters
python -m claimctl.cli timeline contradictions detect --min-confidence 0.7 --max-days 45

# Manually mark a contradiction between two events
python -m claimctl.cli timeline contradictions mark 123 456 "Events describe conflicting completion dates"
```

## Example Workflows

### Using Direct CLI Commands

```bash
# Activate environment
cd ~/Projects/claim-assistant
source venv/bin/activate

# Create and set up a matter
python -m claimctl.cli matter create "Highway Project"

# Ingest documents into the matter
python -m claimctl.cli ingest ~/test-pdfs/*.pdf

# Ask a question with default settings (retrieves top 6 documents)
python -m claimctl.cli ask "Where is Change Order 12 justified?"

# Extract timeline events from all documents
python -m claimctl.cli timeline extract

# View the timeline
python -m claimctl.cli timeline show

# Show financial impact summary
python -m claimctl.cli timeline financials summary

# Export timeline as PDF
python -m claimctl.cli timeline export --open

# Create another matter and switch to it
python -m claimctl.cli matter create "Office Building Project"
python -m claimctl.cli matter switch "Office Building Project"

# Ingest different documents into this matter
python -m claimctl.cli ingest ~/office-project-docs/*.pdf

# Ask a question with more documents for complex queries
python -m claimctl.cli ask "Compare all the change orders related to site conditions" --top-k 10

# Switch back to the first matter
python -m claimctl.cli matter switch "Highway Project"

# Filter results by document type and date range
python -m claimctl.cli ask "What were the approved costs?" --type "ChangeOrder" --from "2024-01-01"

# Sort/filter results
# Type 's' at the prompt

# Ask a follow-up question
# Type 'f' at the prompt

# Export a page to the exhibits folder
# Type 'e' at the prompt

# Open the PDF
# Type 'o' at the prompt

# Export response as PDF with all referenced documents
# Type 'p' at the prompt
```

### Using the Interactive Shell

```bash
# Activate environment and launch the shell
cd ~/Projects/claim-assistant
./run.sh

# Inside the interactive shell:

# Create and set up a matter
matter create "Highway Project"

# Ingest documents (with various options)
ingest ~/test-pdfs/*.pdf
ingest ~/test-pdfs/*.pdf --no-logging  # Disable detailed logging
ingest ~/test-pdfs/*.pdf --batch-size 10

# View ingestion logs
logs list
logs show

# Ask a question
ask Where is Change Order 12 justified?

# Extract and view timeline
timeline extract
timeline show

# View financial summary
timeline financials summary

# Export timeline as PDF
timeline export --open

# Create another matter and switch to it
matter create "Office Building Project"
matter switch "Office Building Project"

# Ingest different documents
ingest ~/office-project-docs/*.pdf

# Ask a complex question
ask Compare all the change orders related to site conditions --top-k 10

# Switch back to the first matter
matter switch "Highway Project"

# Filter results by type and date
ask What were the approved costs? --type ChangeOrder --from 2024-01-01

# Type 'help' for available commands
# Type 'exit' to quit the shell
```