# Document Boundary Detection Tests

This directory contains tests for the document boundary detection system, which identifies logical document boundaries within large PDFs containing multiple documents.

## Test Components

- `test_boundary_detection.py`: Tests the accuracy of the boundary detection algorithm by comparing detected boundaries against ground truth.
- `test_document_segmentation.py`: Tests the document segmentation functionality, including boundary detection and PDF splitting.
- `create_test_dataset.py`: Generates synthetic test data with known document boundaries for evaluation.

## Running Tests

```bash
# Run boundary detection accuracy tests
python -m tests.test_boundary_detection

# Run document segmentation tests
python -m tests.test_document_segmentation

# Generate a new test dataset
python -m tests.create_test_dataset
```

## Test Dataset

The test dataset consists of PDF files containing multiple document types with known boundaries. Each PDF includes:
- Email correspondence
- Change orders
- Meeting minutes
- Daily reports
- Invoices
- Letters
- Submittals
- RFIs

The ground truth for document boundaries is stored in `tests/fixtures/boundary_test_dataset/metadata.json`.

## Performance Metrics

The boundary detection is evaluated using the following metrics:
- **Precision**: Proportion of detected boundaries that match ground truth
- **Recall**: Proportion of ground truth boundaries that were detected
- **F1 Score**: Harmonic mean of precision and recall
- **Mean Distance Error**: Average distance between detected and ground truth boundaries

Current baseline performance:
- Precision: 0.29
- Recall: 0.29
- F1 Score: 0.29

## Improving Performance

To improve boundary detection performance:

1. **Embeddings Refinement**: The system uses OpenAI embeddings to detect semantic shifts. Fine-tuning the embedding model for construction documents could improve performance.

2. **Parameter Tuning**: Adjust the parameters like `threshold_multiplier`, `min_confidence`, `segment_size`, and `segment_stride` based on your specific document characteristics.

3. **Pattern Recognition**: Enhance the boundary indicators in `score_boundary()` function to better match your document types.

4. **Multi-Strategy Approach**: Combine semantic boundaries with layout analysis (page breaks, formatting changes) for better detection.

## Visualizations

Test runs generate boundary detection visualizations in `tests/fixtures/boundary_test_dataset/results/`. These visualizations show:
- Similarity scores between adjacent segments
- Detected boundaries
- Confidence scores
- Evaluation metrics

These visualizations are helpful for understanding detection performance and diagnosing issues.