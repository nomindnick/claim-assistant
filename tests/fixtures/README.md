# Test Fixtures

This directory contains test fixtures for claim-assistant.

## Sample PDFs

Place test PDFs in the `sample_pdfs` directory. These should be representative of the types of documents you'll be analyzing:

- Contract documents
- Change orders
- Email correspondence
- Invoices
- Project schedules
- Photos and diagrams

## golden.yml

The `golden.yml` file maps test questions to expected document results in the format:

```yaml
"Question text": "expected_file.pdf:page_number"
```

This is used for testing the retrieval quality of the system. The test passes if at least 80% of the questions return the expected document in the top results.