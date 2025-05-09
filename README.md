# Construction Claim Assistant CLI

A powerful tool to answer natural-language questions about construction claims and locate relevant evidence in PDF documents with precision.

## Overview

Construction Claim Assistant is designed to help construction professionals, attorneys, and consultants analyze large collections of project documentation to build and substantiate construction claims. The tool processes PDF documents, indexes their content, and allows users to ask natural language questions to find relevant information across the entire document set.

### Key Features

- **Document Ingestion**: Processes both digital and scanned PDFs with automatic OCR fallback
- **Smart Categorization**: Classifies documents by type (emails, change orders, invoices, etc.)
- **Enhanced Metadata Extraction**: Extracts monetary amounts, time periods, contract sections, and work descriptions
- **Advanced Semantic Chunking**: Automatically adapts chunking strategy based on document structure
- **Memory-Optimized Processing**: Efficiently handles very large documents with streaming techniques
- **Natural Language Queries**: Ask questions in plain English about your claim
- **Cross-Encoder Reranking**: Advanced relevance ranking of search results for higher precision
- **Evidence Locator**: Points to exact PDF pages that support your claim
- **Interactive Results**: Open PDFs directly or export pages as exhibits
- **Matter Management**: Work on multiple legal matters simultaneously with isolated document storage
- **Interactive Shell**: Command-line shell with history, auto-completion, and contextual help
- **Chunk Visualization**: Tools to visualize and compare different document chunking methods
- **Ingestion Logging**: Detailed metrics and statistics about document processing and classification
- **Timeline Generation**: Automatically extracts events during ingestion to create a claim chronology with visual display
- **Financial Impact Tracking**: Monitors monetary changes across change orders, payments, and claims
- **Contradiction Detection**: Identifies conflicting information between different project documents
- **Timeline Export**: Exports comprehensive timelines as PDFs with financial analysis
- **Timeline-Query Integration**: Automatically incorporates timeline data into chronology-related questions
- **Low Resource Usage**: Works efficiently on standard laptops (≤3GB RAM)

## Installation

### Prerequisites

- Python 3.10 or higher
- Tesseract OCR engine (for scanned document processing)
- OpenAI API key
- sentence-transformers (automatically installed during setup)

### Quick Install

```bash
git clone https://github.com/yourusername/claim-assistant.git
cd claim-assistant
./setup_env.sh
```

The setup script will:
1. Check for required dependencies
2. Create a virtual environment
3. Install all required Python packages
4. Create a configuration file if needed

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/claim-assistant.git
cd claim-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Configure
cp claimctl.ini.sample ~/.claimctl.ini
# Edit ~/.claimctl.ini to add your OpenAI API key
```

## Configuration

The tool uses a configuration file located at `~/.claimctl.ini`:

```ini
[paths]
DATA_DIR = ./data
INDEX_DIR = ./index

[openai]
API_KEY = your_openai_api_key
MODEL = gpt-4.1-mini
EMBED_MODEL = text-embedding-3-large

[retrieval]
TOP_K = 20
SCORE_THRESHOLD = 0.6
CONTEXT_SIZE = 2000
ANSWER_CONFIDENCE = True
RERANK_ENABLED = True

[chunking]
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
SEMANTIC_CHUNKING = True
HIERARCHICAL_CHUNKING = True
ADAPTIVE_CHUNKING = True
LARGE_DOC_THRESHOLD = 500000
SIMILARITY_THRESHOLD = 0.8

[bm25]
K1 = 1.5
B = 0.75
WEIGHT = 0.3

[project]
DEFAULT_PROJECT = 

[matter]
MATTER_DIR = ./matters
CURRENT_MATTER = 
MATTER_SETTINGS = {}

[timeline]
AUTO_EXTRACT = True
EXTRACT_CONFIDENCE_THRESHOLD = 0.5
EXTRACT_IMPORTANCE_THRESHOLD = 0.3
EXTRACTION_BATCH_SIZE = 10
```

You can also use environment variables to override these settings:
```bash
CLAIMCTL_OPENAI_API_KEY=your_key claimctl ask "Where is the delay mentioned?"
```

## Usage

### Launching the Interactive Shell

The easiest way to get started is to use the interactive shell:

```bash
# Launch the interactive shell
./run.sh

# Or use the command directly
python claim_assistant.py
```

The interactive shell provides:
- Command history and auto-completion
- Visual display of the current matter
- Simplified command syntax
- Contextual help and tab completion

### Ingesting Documents

First, create a matter and then ingest your PDF documents into it:

```bash
# Create a matter to store your documents
claimctl matter create "Highway Project"

# Ingest one or more PDF files into the current matter
claimctl ingest path/to/document1.pdf path/to/document2.pdf

# Use shell expansion to ingest multiple files
claimctl ingest path/to/project/*.pdf

# Ingest into a specific matter
claimctl ingest path/to/project/*.pdf --matter "Office Building Project"

# Control automatic timeline extraction
claimctl ingest path/to/project/*.pdf --timeline-extract  # Enable timeline extraction
claimctl ingest path/to/project/*.pdf --no-timeline-extract  # Disable timeline extraction
```

You can perform the same operations in the interactive shell with simpler syntax.

The ingestion process:
- Extracts text from each page
- Uses OCR for scanned documents
- Saves page images for reference
- Classifies document types
- Generates embeddings for semantic search
- Stores metadata in a local database
- Automatically extracts timeline events for chronology

### Managing Matters

The system allows you to work with multiple legal matters simultaneously:

```bash
# Create a new matter
claimctl matter create "Smith Construction Claim"

# List all matters 
claimctl matter list

# Switch between matters
claimctl matter switch "Jones Project"

# Show matter information
claimctl matter info
```

### Asking Questions

Once your documents are ingested into a matter, you can ask questions:

```bash
# Using the current active matter
claimctl ask "Where is Change Order 12 justified?"

# Specify a different matter
claimctl ask "What caused the delay?" --matter "Jones Project"
```

The output includes:
- A detailed answer synthesized from the relevant documents
- A list of source documents with their relevance scores
- Interactive options to open PDFs or export pages as exhibits

### Output Formats

You can export results in different formats:

```bash
# Output as JSON for programmatic use
claimctl ask "What was the project completion date?" --json

# Output as Markdown for reports
claimctl ask "What caused the delay in foundation work?" --md
```

### Managing Configuration

```bash
# Show current configuration
claimctl config show

# Initialize directories
claimctl config init
```

## Documentation

- [USAGE.md](USAGE.md): Detailed usage guide including advanced features
- [docs/chunking.md](docs/chunking.md): Documentation on advanced document chunking features

## Project Structure

```
claim-assistant/
│
├── data/               # Global data storage
│   ├── raw/            # Original PDFs
│   ├── pages/          # One PNG per PDF page
│   └── cache/          # SHA-256 hashes, OCR txt, etc.
│
├── docs/               # Additional documentation
│   └── chunking.md     # Advanced chunking documentation
│
├── index/              # Global vector index and metadata
│   ├── faiss.idx       # Vector store
│   └── catalog.db      # SQLite metadata
│
├── matters/            # Matter-specific data
│   ├── Matter1/        # Data for a specific matter
│   │   ├── data/       # Matter-specific data storage
│   │   ├── index/      # Matter-specific index
│   │   └── logs/       # Ingestion logs for this matter
│   └── Matter2/        # Another matter
│       ├── data/
│       ├── index/
│       └── logs/
│
├── exhibits/           # Exported page images
│
├── Timeline_Exports/   # Exported timeline PDFs
│
├── claimctl/           # Python package
│   ├── ingest.py       # PDF processing 
│   ├── query.py        # Question answering
│   ├── config.py       # Configuration management
│   ├── semantic_chunking.py # Advanced chunking functionality
│   ├── ingestion_logger.py # Logging and metrics for ingestion
│   ├── timeline.py     # Timeline extraction and management
│   ├── rerank.py       # Cross-encoder reranking functionality
│   └── utils.py        # Utility functions
│
├── test_chunking.py    # Chunking visualization tool
│
└── tests/              # Automated tests
```

## Testing

Run tests to ensure everything is working correctly:

```bash
pytest
```

## License

MIT

## Acknowledgments

- PyMuPDF for PDF processing
- OpenAI for embeddings and text generation
- FAISS for vector search
- Tesseract for OCR processing