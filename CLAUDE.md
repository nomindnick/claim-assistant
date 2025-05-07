# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `poetry install`
- Run CLI: `python -m claimctl.cli` or `./run.sh`
- Run interactive shell: `./run.sh` or `python claim_assistant.py`
- Create matter: `python -m claimctl.cli matter create "My Matter"`
- Switch matter: `python -m claimctl.cli matter switch "My Matter"`

## Interactive Shell Commands
- Launch shell: `./run.sh` or `python claim_assistant.py`
- Get help: Type `help` in the shell
- Create matter: `matter create "My Matter"`
- Switch matter: `matter switch "My Matter"`
- Ingest documents: `ingest path/to/files/*.pdf`
- Ask questions: `ask What caused the delay?`
- Exit shell: `exit` or `quit`

## Test Commands
- All tests: `pytest`
- Single test: `pytest tests/test_retrieval.py::test_recall`
- Verbose: `pytest -v`

## Lint/Format Commands
- Format code: `black .`
- Check format: `black --check .`
- Sort imports: `isort .`
- Type check: `mypy .`
- Lint: `ruff check .`

## Dependencies
- Python 3.10+
- Tesseract OCR for scanned document processing
- OpenAI API key (set in ~/.claimctl.ini)
- PyMuPDF, FAISS, SQLite and other Python libraries (installed via poetry)

## Code Style
- Formatting: Black with 88 character line length
- Imports: Use isort with standard grouping (stdlib, third-party, local)
- Types: Strict typing with Pydantic for data validation
- Naming: snake_case for functions/variables, PascalCase for classes
- Docstrings: Triple-quotes for all modules, classes, and functions
- Error handling: Use specific exceptions with meaningful error messages
- CLI: Built with Typer and Rich for formatted output

## Matter Management
- Always check for active matter with `get_current_matter()` before operations
- Use `--matter` flag when ingesting or querying specific matters
- Configure matter-specific paths in `config.py`
- Document relationships include matter_id for isolation between matters
- Matter files stored in ./matters/{matter_name}/ (excluded from git)

## Key Components

### Document Ingestion
- Processing happens in `claimctl/ingest.py`
- Ingestion pipeline: PDF extraction → OCR → chunking → classification → embedding
- Advanced semantic chunking is implemented in `claimctl/semantic_chunking.py`
- Document classification uses OpenAI models to categorize documents (Email, ChangeOrder, etc.)
- Detailed metrics logging is handled by `claimctl/ingestion_logger.py`
- Large documents (>500K chars) use memory-optimized processing

### Search and Retrieval
- Hybrid search combines vector search and BM25 keyword search
- Initial search retrieves TOP_K * 10 candidates (200 by default)
- Cross-encoder reranking improves search precision
- Top 25 documents passed to the LLM for comprehensive analysis
- Search configuration is handled in the INI file (TOP_K, SCORE_THRESHOLD, etc.)
- Filters available by document type, date range, project name, monetary amount, etc.

### Advanced Chunking
- Four available chunking methods:
  1. Regular: Character-based chunking with attention to natural separators
  2. Semantic: Uses embeddings to identify natural semantic boundaries
  3. Hierarchical: Creates structured representation of documents like contracts
  4. Adaptive: Automatically analyzes document structure and selects optimal method
- Chunking visualization tool in `test_chunking.py`
- See `docs/chunking.md` for detailed explanations

### Model Usage
- Main completion model (`gpt-4o-mini` by default): Answers questions
- Embedding model (`text-embedding-3-large` by default): Generates vector embeddings
- Cross-encoder model: Reranks search results for better precision
- Models can be configured in `~/.claimctl.ini`

### Database Structure
- SQLite database stores documents, pages, chunks, and matter metadata
- FAISS vector index stores embeddings for semantic search
- Each matter has isolated data and index directories
- Diagnostic tools available in `diagnose_db.py` and `fix_database.py`

## Configuration
- Main config file: `~/.claimctl.ini` (from template `claimctl.ini.sample`)
- Key settings include:
  - OpenAI API key and model selection
  - Retrieval parameters (TOP_K, SCORE_THRESHOLD, etc.)
  - Chunking parameters (CHUNK_SIZE, SEMANTIC_CHUNKING, etc.)
  - Matter-specific settings
- Override with environment variables like `CLAIMCTL_OPENAI_API_KEY`

## Recent Features
- Advanced semantic chunking for construction documents
- Detailed ingestion logging system
- Enhanced metadata extraction
- Cross-encoder reranking for improved search precision