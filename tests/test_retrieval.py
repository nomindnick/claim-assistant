"""Basic retrieval tests for claim-assistant."""

import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from claimctl.cli import app
from claimctl.config import get_config
from claimctl.query import search_documents


@pytest.fixture
def fixture_data():
    """Load test fixture data."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    golden_path = fixtures_dir / "golden.yml"

    if not golden_path.exists():
        return {}

    with open(golden_path, "r") as f:
        return yaml.safe_load(f)


def test_recall(fixture_data):
    """Test recall rate against golden set."""
    if not fixture_data:
        pytest.skip("No fixture data available")

    # Skip if no PDFs have been ingested
    config = get_config()
    db_path = Path(config.paths.INDEX_DIR) / "catalog.db"
    if not db_path.exists():
        pytest.skip("No documents ingested")

    # Track metrics
    total_questions = 0
    correct_hits = 0

    for question, expected in fixture_data.items():
        total_questions += 1

        # Search for documents
        chunks, _ = search_documents(question)

        # Check if expected documents are in the results
        found = False
        for chunk in chunks:
            # Format: file:page or just file
            expected_parts = expected.split(":")
            if len(expected_parts) == 2:
                expected_file, expected_page = expected_parts
                if (
                    expected_file in chunk["file_name"]
                    and int(expected_page) == chunk["page_num"]
                ):
                    found = True
                    break
            else:
                if expected in chunk["file_name"]:
                    found = True
                    break

        if found:
            correct_hits += 1

    # Calculate recall rate
    recall = correct_hits / total_questions if total_questions > 0 else 0

    # Test should pass if recall >= 0.8
    assert recall >= 0.8, f"Recall rate too low: {recall:.2f}"


def test_hallucination():
    """Test hallucination rate using nonsense questions."""
    # Skip if no PDFs have been ingested
    config = get_config()
    db_path = Path(config.paths.INDEX_DIR) / "catalog.db"
    if not db_path.exists():
        pytest.skip("No documents ingested")

    # List of nonsense questions that shouldn't match anything
    nonsense_questions = [
        "What is the airspeed velocity of an unladen swallow?",
        "Where can I find unicorn insurance policies?",
        "Does this claim involve time travel technology?",
        "Who makes the best pizza on Jupiter?",
        "How many dragons were included in the contract?",
    ]

    # Track metrics
    total_questions = len(nonsense_questions)
    hallucinations = 0

    for question in nonsense_questions:
        # Search for documents
        chunks, scores = search_documents(question)

        # Check if any document scored above threshold
        matched = any(score > 0.8 for score in scores)

        if matched:
            hallucinations += 1

    # Calculate hallucination rate
    hallucination_rate = hallucinations / total_questions if total_questions > 0 else 0

    # Test should pass if hallucination rate <= 0.15
    assert (
        hallucination_rate <= 0.15
    ), f"Hallucination rate too high: {hallucination_rate:.2f}"
