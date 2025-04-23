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
```

The system will:
1. Find relevant documents based on semantic similarity
2. Generate a comprehensive answer using GPT-4o-mini
3. Display source documents with their relevance scores

### Interactive Commands

After seeing the results, you can interact with the documents:

- Type `o` to open the most relevant PDF
- Type `o 2` to open the second most relevant PDF
- Type `e` to export the most relevant page as an image to the exhibits folder
- Type `e 3` to export the third most relevant page
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

### Configuration Options

You can edit `~/.claimctl.ini` to change:
- Data storage locations
- OpenAI model selection
- Retrieval parameters (number of results, relevance threshold)
- Chunking parameters

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

## Example Workflow

```bash
# Activate environment
cd ~/Projects/claim-assistant
source venv/bin/activate

# Ingest documents
python -m claimctl.cli ingest ~/test-pdfs/*.pdf

# Ask a question
python -m claimctl.cli ask "Where is Change Order 12 justified?"

# Export a page to the exhibits folder (after seeing results)
# Type 'e' at the prompt

# Open the PDF (after seeing results)
# Type 'o' at the prompt
```