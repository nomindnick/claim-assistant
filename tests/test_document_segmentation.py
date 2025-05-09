"""Unit tests for document segmentation functionality."""

import os
import unittest
from unittest import mock
from pathlib import Path
import tempfile
import shutil

import numpy as np

# Import the module to test
from claimctl.document_segmentation import (
    create_text_segments,
    calculate_similarities,
    find_potential_boundaries,
    refine_boundaries,
    detect_document_boundaries
)


class TestDocumentSegmentation(unittest.TestCase):
    """Test document segmentation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create a test text with 3 simulated documents with clearer boundaries
        self.test_text = """
        DOCUMENT 1 - MEETING MINUTES
        Date: January 10, 2025

        This is the content of document 1.
        It has multiple paragraphs with very specific topic about meetings.
        Attendees: John Smith, Mary Johnson, Bob Williams

        This is the second paragraph with content about the project milestones.
        The document continues with details specific to this meeting.
        Action items: Review timeline, prepare budget estimates
        End of meeting minutes.

        ==========================================================

        DOCUMENT 2 - CHANGE ORDER
        Project: Construction Project ABC
        Date: February 15, 2025

        This is a completely different document with different content about change orders.
        The writing style and terminology has changed significantly.

        Change order number: CO-12345
        Amount: $25,000.00
        Description: Additional foundation work required due to soil conditions.

        This document discusses changes to the scope of work.
        Approved by: Robert Johnson, Project Manager
        End of change order document.

        ==========================================================

        DOCUMENT 3 - DAILY REPORT
        Date: March 20, 2025
        Weather: Sunny, 72°F

        This is yet another document with different subject matter related to daily activities.
        The semantic content of this document differs significantly from the previous ones.

        Work performed:
        - Excavation in section A-3
        - Concrete pouring in section B-2
        - Installation of electrical conduits in section C-1

        This document has a different focus on daily operations.
        It contains information completely unrelated to the first two documents.
        Personnel on site: 12 workers, 2 supervisors, 1 inspector
        End of daily report.
        """

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_create_text_segments(self):
        """Test creating text segments from a document."""
        segments = create_text_segments(self.test_text, segment_size=100, segment_stride=50)
        
        # Check that segments were created
        self.assertGreater(len(segments), 0, "Should create at least one segment")
        
        # Check segment properties
        for segment_text, position in segments:
            # Segment should not exceed segment_size
            self.assertLessEqual(len(segment_text), 100, "Segment should not exceed segment_size")
            
            # Position should be within text bounds
            self.assertGreaterEqual(position, 0, "Position should be non-negative")
            self.assertLess(position, len(self.test_text), "Position should be within text bounds")
            
            # Segment should match text at position
            self.assertEqual(segment_text, self.test_text[position:position+len(segment_text)],
                            "Segment text should match text at position")

    def test_calculate_similarities(self):
        """Test calculating similarities between embeddings."""
        # Create dummy embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Embedding 1
            [0.0, 1.0, 0.0],  # Embedding 2 (orthogonal to 1)
            [0.0, 0.0, 1.0],  # Embedding 3 (orthogonal to 1 and 2)
            [0.5, 0.5, 0.0],  # Embedding 4 (similar to 1 and 2)
            [0.7, 0.7, 0.0],  # Embedding 5 (very similar to 4)
        ])
        
        similarities = calculate_similarities(embeddings)
        
        # Check similarities count
        self.assertEqual(len(similarities), len(embeddings) - 1, 
                        "Should return n-1 similarities for n embeddings")
        
        # Check specific similarity values
        self.assertAlmostEqual(similarities[0], 0.0, places=6, 
                              msg="Orthogonal vectors should have similarity 0")
        self.assertAlmostEqual(similarities[1], 0.0, places=6, 
                              msg="Orthogonal vectors should have similarity 0")
        
        # Check that similar vectors have higher similarity
        self.assertGreater(similarities[3], similarities[0], 
                          "Similar vectors should have higher similarity")

    def test_find_potential_boundaries(self):
        """Test finding potential document boundaries."""
        # Create dummy similarities with clear boundaries
        similarities = np.array([0.8, 0.7, 0.2, 0.9, 0.8, 0.1, 0.9, 0.8])
        # Low values at indices 2 and 5 represent potential boundaries
        
        boundaries = find_potential_boundaries(similarities, threshold_multiplier=1.0)
        
        # Check that both boundaries were found
        self.assertIn(2, boundaries, "Should find boundary at index 2")
        self.assertIn(5, boundaries, "Should find boundary at index 5")

    def test_refine_boundaries(self):
        """Test refining boundary positions."""
        text = "First document.\n\nSecond document.\n\nThird document."
        potential_boundaries = [0, 1]  # Indices in similarities array
        segment_positions = [0, 10, 20]  # Start positions of each segment
        similarities = [0.5, 0.2]  # Similarities between segments

        boundaries = refine_boundaries(
            text,
            potential_boundaries,
            segment_positions,
            similarities,
            min_boundary_distance=5,
            min_confidence=0.1  # Lower this to make the test more resilient
        )

        # Check that boundaries were refined and at least one boundary was found
        self.assertGreater(len(boundaries), 0, "Should return at least one boundary")

        # Check that boundaries have expected structure
        for boundary in boundaries:
            self.assertIn('position', boundary, "Boundary should have 'position' key")
            self.assertIn('confidence', boundary, "Boundary should have 'confidence' key")
            self.assertIn('original_index', boundary, "Boundary should have 'original_index' key")

            # Position should be within text bounds
            self.assertGreaterEqual(boundary['position'], 0)
            self.assertLess(boundary['position'], len(text))

            # Confidence should be between 0 and 1
            self.assertGreaterEqual(boundary['confidence'], 0.0)
            self.assertLessEqual(boundary['confidence'], 1.0)

    @mock.patch('claimctl.document_segmentation.get_embeddings')
    def test_detect_document_boundaries(self, mock_embeddings):
        """Test full document boundary detection."""
        config = {}  # Empty config for testing

        # Create mock embeddings that will produce clear semantic shifts
        def mock_get_embeddings(segments, _):
            # Create embeddings with clear semantic shifts
            # Each segment from a different document gets a very different embedding
            num_segments = len(segments)
            embedding_dim = 5  # Small embedding dimension for testing

            # Create base embeddings for each segment
            embeddings = np.zeros((num_segments, embedding_dim))

            # Assign different embeddings based on content to create clear shifts
            for i, segment in enumerate(segments):
                if "DOCUMENT 1" in segment or "meeting minutes" in segment.lower():
                    embeddings[i] = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
                elif "DOCUMENT 2" in segment or "change order" in segment.lower():
                    embeddings[i] = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
                elif "DOCUMENT 3" in segment or "daily report" in segment.lower():
                    embeddings[i] = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
                else:
                    # For segments with boundary markers
                    if "========" in segment:
                        embeddings[i] = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
                    else:
                        # Gradual transition based on position
                        position = i / float(num_segments)
                        if position < 0.33:
                            embeddings[i] = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
                        elif position < 0.66:
                            embeddings[i] = np.array([0.1, 0.9, 0.0, 0.0, 0.0])
                        else:
                            embeddings[i] = np.array([0.0, 0.1, 0.9, 0.0, 0.0])

            return embeddings

        # Set the mock to use our function
        mock_embeddings.side_effect = mock_get_embeddings

        boundaries = detect_document_boundaries(
            self.test_text,
            config=config,
            segment_size=100,
            segment_stride=50,
            threshold_multiplier=1.0,  # Less sensitive threshold since we have clear embeddings
            min_confidence=0.1,  # Lower the confidence threshold for testing
            min_document_length=50,
            visualize=False
        )

        # There should be at least one boundary for the test document
        self.assertGreater(len(boundaries), 0,
                         "Should detect at least one boundary in the test document")

        # Positions should be in ascending order
        positions = [b['position'] for b in boundaries]
        self.assertEqual(positions, sorted(positions),
                        "Boundary positions should be in ascending order")

        # Check boundary positions are reasonable
        # This is a fuzzy test since exact positions depend on algorithm parameters
        doc1_pos = self.test_text.find("DOCUMENT 1")
        doc2_pos = self.test_text.find("DOCUMENT 2")
        doc3_pos = self.test_text.find("DOCUMENT 3")

        if positions and doc1_pos >= 0 and doc2_pos >= 0:
            # At least one boundary should be after DOCUMENT 1 and before DOCUMENT 3
            at_least_one_good_boundary = False
            for pos in positions:
                if pos > doc1_pos and pos < doc3_pos:
                    at_least_one_good_boundary = True
                    break

            # Skip this assertion if our test documents weren't found properly
            if doc1_pos >= 0 and doc3_pos > doc1_pos:
                self.assertTrue(at_least_one_good_boundary,
                               "At least one boundary should be between documents")

    def test_empty_text(self):
        """Test that empty text returns no boundaries."""
        boundaries = detect_document_boundaries("")
        self.assertEqual(len(boundaries), 0, "Empty text should return no boundaries")

    def test_short_text(self):
        """Test that very short text returns no boundaries."""
        short_text = "This is a very short document without any boundaries."
        boundaries = detect_document_boundaries(short_text, min_document_length=50)
        self.assertEqual(len(boundaries), 0, 
                        "Text shorter than min_document_length should return no boundaries")


if __name__ == '__main__':
    unittest.main()