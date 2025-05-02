#!/usr/bin/env python3
"""Test script for the reranking functionality."""

from claimctl.search import search_documents
from claimctl.config import get_config

def test_reranking():
    """Test the reranking functionality by comparing results with and without reranking."""
    # Test query
    test_query = "change order approval for foundation work"

    # Get config
    config = get_config()
    
    # Save original rerank setting
    original_rerank_setting = config.retrieval.RERANK_ENABLED
    
    # Get results without reranking
    config.retrieval.RERANK_ENABLED = False
    chunks_no_rerank, scores_no_rerank = search_documents(test_query)

    print("Results without reranking:")
    for i, (chunk, score) in enumerate(zip(chunks_no_rerank, scores_no_rerank)):
        print(f"{i+1}. {chunk['file_name']} (Page {chunk['page_num']}): {score:.4f}")

    # Get results with reranking
    config.retrieval.RERANK_ENABLED = True
    chunks_reranked, scores_reranked = search_documents(test_query)

    print("\nResults with reranking:")
    for i, (chunk, score) in enumerate(zip(chunks_reranked, scores_reranked)):
        print(f"{i+1}. {chunk['file_name']} (Page {chunk['page_num']}): {score:.4f}")

    # Restore original setting
    config.retrieval.RERANK_ENABLED = original_rerank_setting

if __name__ == "__main__":
    test_reranking()