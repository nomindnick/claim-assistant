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
```

During ingestion, the system:
- Extracts text from each page
- Runs OCR if needed for scanned content
- Saves page images for reference
- Classifies document types (Email, ChangeOrder, Invoice, etc.)
- Generates embeddings for semantic search
- Stores everything in the database

### Asking Questions

Once your documents are ingested, you can ask natural language questions:

```bash
python -m claimctl.cli ask "Where is Change Order 12 justified?"

# Specify the number of documents to use (default is 6)
python -m claimctl.cli ask "Where is Change Order 12 justified?" --top-k 10
# or using the shorter flag
python -m claimctl.cli ask "Where is Change Order 12 justified?" -k 10
```

The system will:
1. Find relevant documents based on semantic similarity (by default, the top 6 documents)
2. Generate a comprehensive answer using GPT-4o-mini
3. Display source documents with their relevance scores

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
  - TOP_K: Number of documents to retrieve (default is 6)
  - SCORE_THRESHOLD: Minimum similarity score for documents
  - CONTEXT_SIZE: Character limit per chunk for context window
  - ANSWER_CONFIDENCE: Whether to include confidence indicators
- Chunking parameters (CHUNK_SIZE, CHUNK_OVERLAP)
- BM25 search parameters (K1, B, WEIGHT)
- Project settings (DEFAULT_PROJECT)
- Matter settings (MATTER_DIR, CURRENT_MATTER)

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

## Example Workflow

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