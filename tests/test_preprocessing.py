"""Integration tests for document preprocessing functionality."""

import os
import unittest
from pathlib import Path
import tempfile
import shutil
import json

from unittest import mock

# Import modules to test
from claimctl.preprocessing import (
    preprocess_pdf,
    process_large_pdf,
    prepare_db_schema,
    should_preprocess_pdf,
    store_document_relationship,
    get_derived_documents,
    get_original_document
)


class TestPreprocessing(unittest.TestCase):
    """Test document preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock config
        self.config = {
            'document_segmentation': {
                'ENABLED': True,
                'THRESHOLD_MULTIPLIER': 1.5,
                'MIN_CONFIDENCE': 0.3,
                'MIN_DOCUMENT_LENGTH': 1000,
                'MIN_BOUNDARY_DISTANCE': 2000,
                'PAGES_THRESHOLD': 50,
                'SIZE_THRESHOLD': 10000000,
                'VISUALIZE': False
            }
        }
        
        # Mock the database connection
        self.db_patcher = mock.patch('claimctl.preprocessing.get_db_connection')
        self.mock_db = self.db_patcher.start()
        self.mock_cursor = mock.MagicMock()
        self.mock_db.return_value.cursor.return_value = self.mock_cursor
        self.mock_cursor.lastrowid = 1  # For testing store_document_relationship

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)
        
        # Stop mocks
        self.db_patcher.stop()

    @mock.patch('claimctl.preprocessing.calculate_file_hash')
    @mock.patch('claimctl.preprocessing.process_pdf_for_segmentation')
    def test_preprocess_pdf(self, mock_process, mock_hash):
        """Test preprocessing a PDF."""
        # Mock the return value of process_pdf_for_segmentation
        mock_process.return_value = {
            'original_pdf': '/path/to/test.pdf',
            'text_length': 10000,
            'documents_found': 2,
            'documents': [
                {
                    'path': '/path/to/output/test_doc1.pdf',
                    'start_page': 0,
                    'end_page': 10,
                    'page_count': 11,
                    'doc_type': 'contract',
                    'confidence': 0.8
                },
                {
                    'path': '/path/to/output/test_doc2.pdf',
                    'start_page': 11,
                    'end_page': 20,
                    'page_count': 10,
                    'doc_type': 'email',
                    'confidence': 0.9
                }
            ]
        }

        # Mock the file hash calculation to avoid file not found errors
        mock_hash.return_value = 'mock_hash_value'

        result = preprocess_pdf('/path/to/test.pdf', self.output_dir, self.config)

        # Check that the function was called correctly
        mock_process.assert_called_once()
        args, kwargs = mock_process.call_args
        self.assertEqual(kwargs['pdf_path'], '/path/to/test.pdf')
        self.assertEqual(kwargs['output_dir'], self.output_dir)
        # Don't check exact config contents as they're transformed in the function

        # Check the result
        self.assertEqual(result['documents_found'], 2)
        self.assertEqual(len(result['documents']), 2)
        self.assertEqual(result['documents'][0]['doc_type'], 'contract')
        self.assertEqual(result['documents'][1]['doc_type'], 'email')

        # Check that file hashes were added
        self.assertIn('original_pdf_hash', result)
        self.assertIn('hash', result['documents'][0])
        self.assertIn('hash', result['documents'][1])

    @mock.patch('claimctl.preprocessing.calculate_file_hash')
    @mock.patch('claimctl.preprocessing.preprocess_pdf')
    def test_process_large_pdf(self, mock_preprocess, mock_hash):
        """Test processing a large PDF."""
        # Mock the return value of preprocess_pdf
        mock_preprocess.return_value = {
            'original_pdf_hash': 'abc123',
            'documents_found': 2,
            'documents': [
                {
                    'path': '/path/to/output/test_doc1.pdf',
                    'start_page': 0,
                    'end_page': 10,
                    'doc_type': 'contract',
                    'confidence': 0.8,
                    'hash': 'def456'
                },
                {
                    'path': '/path/to/output/test_doc2.pdf',
                    'start_page': 11,
                    'end_page': 20,
                    'doc_type': 'email',
                    'confidence': 0.9,
                    'hash': 'ghi789'
                }
            ]
        }

        # Mock the file hash calculation
        mock_hash.side_effect = lambda path: {
            '/path/to/test.pdf': 'abc123',
            '/path/to/output/test_doc1.pdf': 'def456',
            '/path/to/output/test_doc2.pdf': 'ghi789'
        }[path]

        result = process_large_pdf('/path/to/test.pdf', self.output_dir, 'test_matter', self.config)

        # Check that preprocess_pdf was called correctly
        mock_preprocess.assert_called_once()

        # In the current implementation, execute is called multiple times:
        # - Once for prepare_db_schema (CREATE TABLE)
        # - Twice for indices creation
        # - Twice for store_document_relationship (once per document)
        # So we expect at least 5 calls in total
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 5)

        # Check the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], '/path/to/output/test_doc1.pdf')
        self.assertEqual(result[1], '/path/to/output/test_doc2.pdf')

    def test_should_preprocess_pdf(self):
        """Test the logic for determining if a PDF should be preprocessed."""
        with mock.patch('os.path.getsize') as mock_size, \
             mock.patch('fitz.open') as mock_open:
            
            # Set up mocks for a large PDF by page count
            mock_size.return_value = 5000000  # 5MB
            mock_doc = mock.MagicMock()
            mock_doc.__len__.return_value = 60  # 60 pages
            mock_open.return_value = mock_doc
            
            # Test a PDF that's large by page count
            result = should_preprocess_pdf('/path/to/large_by_pages.pdf', self.config)
            self.assertTrue(result, "Should preprocess PDF with many pages")
            
            # Reset mocks for a large PDF by file size
            mock_size.return_value = 15000000  # 15MB
            mock_doc.__len__.return_value = 20  # 20 pages
            
            # Test a PDF that's large by file size
            result = should_preprocess_pdf('/path/to/large_by_size.pdf', self.config)
            self.assertTrue(result, "Should preprocess PDF with large file size")
            
            # Reset mocks for a small PDF
            mock_size.return_value = 2000000  # 2MB
            mock_doc.__len__.return_value = 10  # 10 pages
            
            # Test a small PDF
            result = should_preprocess_pdf('/path/to/small.pdf', self.config)
            self.assertFalse(result, "Should not preprocess small PDF")

    def test_prepare_db_schema(self):
        """Test preparing the database schema."""
        prepare_db_schema()
        
        # Check that the correct SQL statements were executed
        calls = self.mock_cursor.execute.call_args_list
        self.assertEqual(len(calls), 3, "Should execute 3 SQL statements")
        
        # Check that CREATE TABLE was called
        create_call = calls[0]
        self.assertIn("CREATE TABLE IF NOT EXISTS document_relationships", 
                     create_call[0][0], "Should create document_relationships table")
        
        # Check that CREATE INDEX statements were called
        index_calls = calls[1:]
        self.assertIn("CREATE INDEX IF NOT EXISTS", index_calls[0][0][0], "Should create indices")
        self.assertIn("CREATE INDEX IF NOT EXISTS", index_calls[1][0][0], "Should create indices")

    def test_store_document_relationship(self):
        """Test storing document relationships in the database."""
        rel_id = store_document_relationship(
            matter_id='test_matter',
            original_pdf='/path/to/original.pdf',
            derived_pdf='/path/to/derived.pdf',
            rel_type='segment',
            start_page=0,
            end_page=10,
            doc_type='contract',
            confidence=0.8,
            original_hash='abc123',
            derived_hash='def456'
        )
        
        # Check the result
        self.assertEqual(rel_id, 1, "Should return the ID of the inserted relationship")
        
        # Check that the correct SQL was executed
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("INSERT INTO document_relationships", call_args[0], 
                     "Should insert into document_relationships table")
        
        # Check that the parameters were passed correctly
        params = call_args[1]
        self.assertEqual(params[0], 'test_matter', "Matter ID should be correct")
        self.assertEqual(params[1], '/path/to/original.pdf', "Original PDF path should be correct")
        self.assertEqual(params[2], '/path/to/derived.pdf', "Derived PDF path should be correct")
        self.assertEqual(params[5], 'segment', "Relationship type should be correct")
        self.assertEqual(params[6], 0, "Start page should be correct")
        self.assertEqual(params[7], 10, "End page should be correct")
        self.assertEqual(params[8], 'contract', "Document type should be correct")
        self.assertEqual(params[9], 0.8, "Confidence should be correct")

    def test_get_derived_documents(self):
        """Test getting derived documents for an original PDF."""
        # Mock database query result
        self.mock_cursor.fetchall.return_value = [
            (1, '/path/to/derived1.pdf', 'segment', 0, 10, 'contract', 0.8, '2023-01-01 12:00:00'),
            (2, '/path/to/derived2.pdf', 'segment', 11, 20, 'email', 0.9, '2023-01-01 12:05:00')
        ]
        
        result = get_derived_documents('test_matter', '/path/to/original.pdf')
        
        # Check the result
        self.assertEqual(len(result), 2, "Should return 2 derived documents")
        self.assertEqual(result[0]['id'], 1, "First document ID should be correct")
        self.assertEqual(result[0]['path'], '/path/to/derived1.pdf', "First document path should be correct")
        self.assertEqual(result[1]['document_type'], 'email', "Second document type should be correct")
        
        # Check that the correct SQL was executed
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("SELECT", call_args[0], "Should select from document_relationships table")
        self.assertIn("WHERE matter_id = ? AND original_pdf_path = ?", call_args[0],
                     "Should filter by matter_id and original_pdf_path")

    def test_get_original_document(self):
        """Test getting the original document for a derived PDF."""
        # Mock database query result
        self.mock_cursor.fetchone.return_value = (
            1, '/path/to/original.pdf', 'segment', 0, 10, 'contract', 0.8, '2023-01-01 12:00:00'
        )
        
        result = get_original_document('test_matter', '/path/to/derived1.pdf')
        
        # Check the result
        self.assertIsNotNone(result, "Should return original document info")
        self.assertEqual(result['id'], 1, "Original document ID should be correct")
        self.assertEqual(result['path'], '/path/to/original.pdf', "Original document path should be correct")
        self.assertEqual(result['document_type'], 'contract', "Original document type should be correct")
        
        # Check that the correct SQL was executed
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("SELECT", call_args[0], "Should select from document_relationships table")
        self.assertIn("WHERE matter_id = ? AND derived_pdf_path = ?", call_args[0],
                     "Should filter by matter_id and derived_pdf_path")


if __name__ == '__main__':
    unittest.main()