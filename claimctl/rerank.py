"""Reranking module for improving search result relevance."""

from typing import Any, Dict, List, Tuple, Optional

from rich.console import Console
console = Console()


def rerank_with_cross_encoder(
    query: str, 
    chunks: List[Dict[str, Any]], 
    scores: List[float],
    top_k: Optional[int] = None,
    llm_document_count: int = 50
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Rerank search results using a cross-encoder model.
    
    Args:
        query: The user's query
        chunks: List of document chunks from initial retrieval
        scores: Initial relevance scores
        top_k: Number of results to return after reranking
        llm_document_count: Number of documents to pass to the LLM (default 25)
        
    Returns:
        Tuple of (reranked_chunks, reranked_scores)
    """
    if not chunks:
        return chunks, scores
        
    if top_k is None:
        top_k = len(chunks)
    
    # Use llm_document_count for number of documents to return
    top_k = min(llm_document_count, len(chunks))
    console.log(f"Reranker received {len(chunks)} chunks, will return up to {top_k} documents")
    
    try:
        # Import here to avoid dependencies if not used
        from sentence_transformers import CrossEncoder
        
        console.log("Loading cross-encoder model...")
        # Choose a lightweight cross-encoder model
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        console.log("Cross-encoder model loaded successfully")
        
        # Prepare query-document pairs
        query_doc_pairs = [(query, chunk["text"][:512]) for chunk in chunks]
        
        # Get relevance scores from cross-encoder
        console.log(f"Running prediction on {len(query_doc_pairs)} document pairs...")
        try:
            rerank_scores = model.predict(query_doc_pairs)
            console.log(f"Prediction successful, got {len(rerank_scores)} scores")
        except Exception as e:
            console.log(f"[bold red]Error during model prediction: {str(e)}")
            return chunks[:top_k], [1.0] * min(top_k, len(chunks))  # Return original chunks with dummy scores
        
        # Create combined results
        results = list(zip(chunks, rerank_scores))
        
        # Sort by cross-encoder score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        reranked_chunks = [item[0] for item in results[:top_k]]
        reranked_scores = [item[1] for item in results[:top_k]]
        
        console.log(f"Reranked {len(chunks)} results using cross-encoder, returning {len(reranked_chunks)} documents to LLM")
        return reranked_chunks, reranked_scores
        
    except Exception as e:
        console.log(f"[bold red]Error during reranking: {str(e)}")
        console.log("[yellow]Falling back to original search results")
        return chunks, scores