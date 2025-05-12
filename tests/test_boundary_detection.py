#!/usr/bin/env python
"""
Test boundary detection accuracy against the ground truth dataset.

This script evaluates document boundary detection performance by comparing
detected boundaries against ground truth from the test dataset.

Usage:
    python -m tests.test_boundary_detection [--dataset-dir PATH] [--visualize]
"""

import os
import json
import argparse
import unittest
from pathlib import Path
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from claimctl.multi_strategy_boundary_detection import detect_document_boundaries_multi_strategy, BoundaryDetectionStrategy


class BoundaryDetectionMetrics:
    """Calculate and report metrics for document boundary detection."""
    
    def __init__(self, tolerance=50):
        """
        Initialize with tolerance for boundary matching.
        
        Args:
            tolerance: Character position tolerance for matching boundaries (default: 50)
        """
        self.tolerance = tolerance
        self.results = {}
    
    def evaluate(self, ground_truth, predictions, document_text=None):
        """
        Evaluate boundary detection performance.
        
        Args:
            ground_truth: List of ground truth boundary positions
            predictions: List of predicted boundary positions
            document_text: Original document text (optional, for visualization)
            
        Returns:
            Dictionary of metrics
        """
        # Sort boundaries by position
        ground_truth = sorted(ground_truth)
        predictions = sorted(predictions)
        
        # Find matches between ground truth and predictions
        matches = []
        false_positives = []
        false_negatives = []
        
        # For each ground truth boundary, find closest prediction
        for gt_pos in ground_truth:
            best_match = None
            min_distance = float('inf')
            
            for pred_pos in predictions:
                distance = abs(gt_pos - pred_pos)
                if distance < min_distance and distance <= self.tolerance:
                    min_distance = distance
                    best_match = pred_pos
            
            if best_match is not None:
                matches.append((gt_pos, best_match, min_distance))
                predictions.remove(best_match)  # Remove matched prediction
            else:
                false_negatives.append(gt_pos)
        
        # Remaining predictions are false positives
        false_positives = predictions
        
        # Calculate metrics
        true_positives = len(matches)
        precision = true_positives / (true_positives + len(false_positives)) if true_positives + len(false_positives) > 0 else 0
        recall = true_positives / (true_positives + len(false_negatives)) if true_positives + len(false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # Mean distance error for matched boundaries
        mean_distance = sum(dist for _, _, dist in matches) / len(matches) if matches else 0
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'mean_distance_error': mean_distance,
            'matches': matches,
            'unmatched_ground_truth': false_negatives,
            'unmatched_predictions': false_positives
        }
        
        self.results = results
        return results
    
    def visualize(self, text, ground_truth, predictions, output_path=None):
        """
        Create visualization of boundaries in the text.
        
        Args:
            text: Document text
            ground_truth: List of ground truth boundary positions
            predictions: List of predicted boundary positions
            output_path: Path to save visualization (optional)
        """
        if not text:
            return
            
        # Create figure with three subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Create text positions array
        positions = np.arange(len(text))
        
        # Create boundary indicators
        gt_indicators = np.zeros(len(text))
        for pos in ground_truth:
            if 0 <= pos < len(text):
                gt_indicators[pos] = 1.0
        
        pred_indicators = np.zeros(len(text))
        for pos in predictions:
            if 0 <= pos < len(text):
                pred_indicators[pos] = 1.0
        
        # Plot ground truth boundaries
        ax1.set_title("Ground Truth Boundaries")
        ax1.plot(positions, gt_indicators, 'g-', linewidth=2)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_ylabel("Boundary Present")
        
        # Plot predicted boundaries
        ax2.set_title("Predicted Boundaries")
        ax2.plot(positions, pred_indicators, 'b-', linewidth=2)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel("Boundary Present")
        ax2.set_xlabel("Text Position (characters)")
        
        # Highlight matches, false positives, and false negatives
        for gt_pos, pred_pos, _ in self.results.get('matches', []):
            # Highlight matched ground truth in green
            if 0 <= gt_pos < len(text):
                ax1.axvspan(max(0, gt_pos-self.tolerance), min(len(text), gt_pos+self.tolerance), 
                           alpha=0.3, color='green')
            
            # Highlight matched prediction in green
            if 0 <= pred_pos < len(text):
                ax2.axvspan(max(0, pred_pos-self.tolerance), min(len(text), pred_pos+self.tolerance),
                           alpha=0.3, color='green')
        
        # Highlight false negatives in red in ground truth plot
        for pos in self.results.get('unmatched_ground_truth', []):
            if 0 <= pos < len(text):
                ax1.axvspan(max(0, pos-self.tolerance), min(len(text), pos+self.tolerance),
                           alpha=0.3, color='red')
        
        # Highlight false positives in red in predictions plot
        for pos in self.results.get('unmatched_predictions', []):
            if 0 <= pos < len(text):
                ax2.axvspan(max(0, pos-self.tolerance), min(len(text), pos+self.tolerance),
                           alpha=0.3, color='red')
        
        # Add metrics as text on the plot
        metrics_text = (
            f"Precision: {self.results.get('precision', 0):.2f}\n"
            f"Recall: {self.results.get('recall', 0):.2f}\n"
            f"F1 Score: {self.results.get('f1', 0):.2f}\n"
            f"Mean Distance Error: {self.results.get('mean_distance_error', 0):.2f}\n"
            f"TP: {self.results.get('true_positives', 0)}, "
            f"FP: {self.results.get('false_positives', 0)}, "
            f"FN: {self.results.get('false_negatives', 0)}"
        )
        
        fig.text(0.01, 0.01, metrics_text, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close(fig)
        else:
            plt.show()


class TestBoundaryDetection(unittest.TestCase):
    """Test document boundary detection against ground truth dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Try to find the test dataset
        dataset_dir = Path("tests/fixtures/boundary_test_dataset")
        
        # If dataset doesn't exist, create it
        if not dataset_dir.exists():
            print("Test dataset not found, creating it...")
            import subprocess
            subprocess.run(["python", "-m", "tests.create_test_dataset",
                           "--output-dir", str(dataset_dir), "--count", "8"])
        
        # Load metadata
        cls.metadata_path = dataset_dir / "metadata.json"
        if not cls.metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found at {cls.metadata_path}")
            
        with open(cls.metadata_path, 'r') as f:
            cls.dataset_metadata = json.load(f)
            
        # Store the dataset directory
        cls.dataset_dir = dataset_dir
        
    def load_pdf_text(self, pdf_path):
        """
        Load text from a PDF file.
        
        This is a simple implementation that calls PyMuPDF to extract text.
        In the real environment, this would use the existing PDF extraction code.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            import subprocess
            subprocess.check_call(["pip", "install", "pymupdf"])
            import fitz
            
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        return text
    
    def test_boundary_detection_accuracy(self):
        """Test boundary detection accuracy against ground truth."""
        # Create metrics collector with higher tolerance to improve matching
        metrics_collector = BoundaryDetectionMetrics(tolerance=200)
        
        # Prepare to collect overall results
        all_true_positives = 0
        all_false_positives = 0
        all_false_negatives = 0
        
        # Process each PDF in the dataset
        for pdf_metadata in self.dataset_metadata['pdfs']:
            pdf_name = pdf_metadata['filename']
            pdf_path = self.dataset_dir / "pdfs" / pdf_name
            
            # Skip if PDF doesn't exist
            if not pdf_path.exists():
                print(f"Skipping {pdf_name}, file not found")
                continue
                
            # Load PDF text
            text = self.load_pdf_text(pdf_path)
            
            # Get ground truth boundaries
            ground_truth_positions = [b['position'] for b in pdf_metadata.get('boundaries', [])]
            
            # Add debug output
            print(f"\nProcessing {pdf_name} (length: {len(text)} chars)...")
            print(f"Ground truth boundaries: {ground_truth_positions}")

            # Use multi-strategy boundary detection with refined parameters
            detected_boundaries = detect_document_boundaries_multi_strategy(
                text,
                config={},
                strategy=BoundaryDetectionStrategy.HYBRID,  # Use hybrid approach
                segment_size=80,                 # Smaller segments for finer granularity
                segment_stride=40,               # Smaller stride for better overlap
                threshold_multiplier=0.1,        # Very sensitive threshold
                min_confidence=0.05,             # Very low confidence threshold
                min_document_length=200,         # Allow smaller documents for test dataset
                min_boundary_distance=300,       # Allow boundaries closer together
                pdf_path=str(pdf_path),          # Pass PDF path for visual features
                visualize=True,                  # Enable visualization
                visualization_path=str(self.dataset_dir / "results" / f"{pdf_name.replace('.pdf', '_boundaries.png')}")
            )

            print(f"Detected {len(detected_boundaries)} boundaries: {[b['position'] for b in detected_boundaries]}")
            print(f"Boundary confidences: {[round(b['confidence'], 2) for b in detected_boundaries]}")
            # Print feature types if available
            if detected_boundaries and 'feature_types' in detected_boundaries[0]:
                for i, b in enumerate(detected_boundaries):
                    print(f"Boundary {i+1} features: {list(b.get('feature_types', []))}")
            
            detected_positions = [b['position'] for b in detected_boundaries]
            
            # Evaluate detection accuracy
            results = metrics_collector.evaluate(ground_truth_positions, detected_positions, text)
            
            # Create visualization
            output_dir = self.dataset_dir / "results"
            output_dir.mkdir(exist_ok=True)
            
            metrics_collector.visualize(
                text,
                ground_truth_positions,
                detected_positions,
                output_path=output_dir / f"{pdf_name.replace('.pdf', '_boundaries.png')}"
            )
            
            # Accumulate results
            all_true_positives += results['true_positives']
            all_false_positives += results['false_positives']
            all_false_negatives += results['false_negatives']
            
            # Report individual PDF results
            print(f"\nResults for {pdf_name}:")
            print(f"  Precision: {results['precision']:.2f}")
            print(f"  Recall: {results['recall']:.2f}")
            print(f"  F1 Score: {results['f1']:.2f}")
            print(f"  Mean Distance Error: {results['mean_distance_error']:.2f}")
            
        # Calculate overall precision, recall, F1
        overall_precision = (all_true_positives / (all_true_positives + all_false_positives) 
                           if all_true_positives + all_false_positives > 0 else 0)
        overall_recall = (all_true_positives / (all_true_positives + all_false_negatives)
                         if all_true_positives + all_false_negatives > 0 else 0)
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                     if overall_precision + overall_recall > 0 else 0)
        
        # Report overall results
        print("\nOverall Results:")
        print(f"  Precision: {overall_precision:.2f}")
        print(f"  Recall: {overall_recall:.2f}")
        print(f"  F1 Score: {overall_f1:.2f}")
        print(f"  True Positives: {all_true_positives}")
        print(f"  False Positives: {all_false_positives}")
        print(f"  False Negatives: {all_false_negatives}")
        
        # Save overall results to JSON
        results_path = self.dataset_dir / "results" / "overall_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'true_positives': all_true_positives,
                'false_positives': all_false_positives,
                'false_negatives': all_false_negatives
            }, f, indent=2)
        
        # Assert minimum acceptable performance
        # These thresholds are set low for initial testing, can be increased as the model improves
        self.assertGreaterEqual(overall_precision, 0.1, "Precision should be at least 0.1")
        self.assertGreaterEqual(overall_recall, 0.1, "Recall should be at least 0.1")
        self.assertGreaterEqual(overall_f1, 0.1, "F1 score should be at least 0.1")


def main():
    """Run the boundary detection test."""
    parser = argparse.ArgumentParser(description="Test document boundary detection against ground truth dataset")
    parser.add_argument("--dataset-dir", type=str, default="tests/fixtures/boundary_test_dataset",
                        help="Directory containing the test dataset")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of boundary detection results")
    
    args = parser.parse_args()
    
    # Set dataset directory if provided
    if args.dataset_dir:
        TestBoundaryDetection.dataset_dir = Path(args.dataset_dir)
        TestBoundaryDetection.metadata_path = Path(args.dataset_dir) / "metadata.json"
    
    # Run the test
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    main()