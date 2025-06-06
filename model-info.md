# Model Configuration in Claim Assistant

The application uses OpenAI models in three places, which can be configured in your `~/.claimctl.ini` file. These settings apply globally but are used in the context of the current active matter:

## Models Used

1. **Main Completion Model** (`MODEL` in config)
   - Default: `gpt-4.1-mini`
   - Used for: Answering questions based on retrieved documents
   - Location in code: `query.py` line ~127
   - Purpose: Generates comprehensive answers to user questions

2. **Document Classifier Model** (`MODEL` in config)
   - Default: `gpt-4.1-mini`
   - Used for: Classifying document types during ingestion
   - Location in code: `ingest.py` line ~102
   - Purpose: Determines if a document is an Email, ChangeOrder, Invoice, etc.

3. **Timeline Extractor Model** (`MODEL` in config)
   - Default: `gpt-4.1-mini`
   - Used for: Extracting timeline events during ingestion
   - Location in code: `timeline.py` line ~142
   - Purpose: Identifies and extracts significant timeline events from documents
   - Also used in query.py for integrating timeline data into question answering

4. **Embedding Model** (`EMBED_MODEL` in config)
   - Default: `text-embedding-3-large`
   - Used for: Generating vector embeddings for text search
   - Location in code: `ingest.py` line ~177
   - Purpose: Creates semantic vector representations for search

## Updating Model Configuration

You can update the models by editing your `~/.claimctl.ini` file:

```ini
[openai]
API_KEY = your-api-key
MODEL = gpt-4.1-mini  # Change to any OpenAI completion model
EMBED_MODEL = text-embedding-3-large  # Change to any OpenAI embedding model

[matter]
MATTER_DIR = ./matters  # Parent directory for all matters
CURRENT_MATTER = Smith Construction Claim  # Currently active matter
```

Or by setting environment variables:

```bash
CLAIMCTL_OPENAI_MODEL=gpt-4 CLAIMCTL_OPENAI_EMBED_MODEL=text-embedding-3-large python -m claimctl.cli ask "..."
```

## Model Warning Messages

If you see "model not found" warnings, it typically means:

1. The embedding model name might need updating (currently using `text-embedding-3-large`)

Make sure the models you specify in the config are valid and available in your OpenAI account. The recommended models are `gpt-4.1-2025-04-14` (powerful but more expensive) and `gpt-4.1-mini` (more economical).