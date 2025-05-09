# Advanced Document Chunking

This document explains the advanced chunking capabilities in claim-assistant.

## Chunking Methods

The system supports four different chunking methods:

1. **Regular Chunking** - Simple character-based chunking that splits text based on character count and preferred separators.
2. **Semantic Chunking** - Uses embeddings to identify natural semantic boundaries in text.
3. **Hierarchical Chunking** - Preserves document structure with a hierarchical approach for structured documents like contracts.
4. **Adaptive Chunking** - Automatically detects document structure and selects the optimal chunking method.

## Configuring Chunking

Chunking options can be configured in your `.claimctl.ini` file:

```ini
[chunking]
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
SEMANTIC_CHUNKING = True
HIERARCHICAL_CHUNKING = True
ADAPTIVE_CHUNKING = True
LARGE_DOC_THRESHOLD = 500000
SIMILARITY_THRESHOLD = 0.8
```

### Options Explained

- `CHUNK_SIZE`: Maximum size of chunks in characters
- `CHUNK_OVERLAP`: Overlap between chunks in characters
- `SEMANTIC_CHUNKING`: Enable semantic chunking (uses embeddings)
- `HIERARCHICAL_CHUNKING`: Enable hierarchical chunking for structured documents
- `ADAPTIVE_CHUNKING`: Enable automatic detection of document structure
- `LARGE_DOC_THRESHOLD`: Character threshold for activating large document processing
- `SIMILARITY_THRESHOLD`: Threshold for detecting duplicate chunks (0.0-1.0)

## Command Line Options

You can override chunking settings when ingesting documents:

```bash
python -m claimctl.cli ingest path/to/documents/*.pdf --adaptive-chunking
```

Available flags:

- `--semantic-chunking/--no-semantic-chunking`: Toggle semantic chunking
- `--hierarchical-chunking/--no-hierarchical-chunking`: Toggle hierarchical chunking
- `--adaptive-chunking/--no-adaptive-chunking`: Toggle adaptive chunking
- `--timeline-extract/--no-timeline-extract`: Toggle automatic timeline event extraction during ingestion

## Chunk Visualization Tool

A visualization tool is included to help understand how different chunking methods affect document partitioning:

```bash
# Compare all chunking methods on a PDF
python test_chunking.py compare path/to/document.pdf --page 0

# Visualize a single chunking method
python test_chunking.py visualize path/to/document.pdf --method adaptive --page 0
```

The visualization tool provides:

1. Visual representation of chunks with color coding
2. Highlighted view showing chunk boundaries in the original text
3. Raw text view showing individual chunks
4. Search functionality to find text across chunks

## Memory Optimization

Large documents (over 500,000 characters) are automatically processed using memory-optimized streaming:

1. The document is segmented into manageable parts
2. Each segment is chunked using the appropriate method
3. Duplicate chunks at segment boundaries are eliminated
4. Progress is tracked throughout the process

This allows processing very large documents without excessive memory usage.

## How Adaptive Chunking Works

Adaptive chunking automatically analyzes document structure using:

1. **Rule-based detection** - Identifies markers like section headings, numbered lists, and other structural elements
2. **Statistical analysis** - Analyzes paragraph length distribution and consistency
3. **AI-based detection** - Uses a language model to detect document structure type

Based on this analysis, the system selects the most appropriate chunking method:

- **Hierarchical chunking** for highly structured documents (contracts, specifications)
- **Semantic chunking** for narrative text with natural paragraph breaks
- **Regular chunking** for tabular data or documents without clear structure

## Relationship to Timeline Extraction

The chunking method used can significantly affect the quality of timeline events extracted from documents:

- **Hierarchical chunking** preserves section relationships which helps with contextual understanding of events described in contracts and formal documents
- **Semantic chunking** ensures that related information about a single event stays together, even across paragraph boundaries
- **Document-aware processing** during timeline extraction further enhances event extraction by maintaining awareness of document context beyond individual chunks

For optimal timeline extraction, use:

```bash
# During ingestion
python -m claimctl.cli ingest path/to/documents/*.pdf --adaptive-chunking

# For manual timeline extraction
python -m claimctl.cli timeline extract --document-aware
```