"""Search functionality for claim-assistant.

This module implements various search algorithms including:
1. Vector-based semantic search
2. BM25 keyword search
3. Hybrid search (combination of vector and keyword)
4. Metadata filtering
"""

import math
import re
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .config import get_config
from .database import (
    get_chunks_by_metadata,
    get_top_chunks_by_similarity,
)
from .rerank import rerank_with_cross_encoder
from .utils import console


class BM25:
    """BM25 implementation for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize with BM25 parameters.

        Args:
            k1: Term saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase and split on non-alphanumeric chars
        tokens = re.findall(r"\w+", text.lower())
        # Remove tokens less than 3 chars (usually not meaningful)
        return [token for token in tokens if len(token) > 2]

    def fit(self, corpus: List[str]) -> None:
        """Fit BM25 parameters to the corpus."""
        self.corpus_size = len(corpus)
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = (
            sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
        )

        # Calculate document frequencies
        df = {}
        for doc in self.tokenized_corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        # Calculate inverse document frequencies
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search the corpus for relevant documents.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_idx, score) tuples
        """
        if not self.corpus_size:
            return []

        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for doc_idx, doc in enumerate(self.tokenized_corpus):
            doc_len = self.doc_lens[doc_idx]

            # Skip empty documents
            if doc_len == 0:
                continue

            # Count term frequencies in the document
            doc_term_freqs = Counter(doc)

            # Calculate BM25 score for this document
            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue

                # Term frequency in document
                freq = doc_term_freqs.get(token, 0)
                if freq == 0:
                    continue

                # BM25 scoring formula
                numerator = self.idf[token] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_len
                )
                score += numerator / denominator

            scores[doc_idx] = score

        # Get top_k results
        results = [(idx, score) for idx, score in enumerate(scores) if score > 0]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 5,
    metadata_filters: Optional[Dict[str, Any]] = None,
    vector_weight: float = 0.7,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Perform hybrid search combining vector and keyword-based approaches.

    Args:
        query: Query string
        top_k: Number of results to return
        metadata_filters: Optional filters for metadata (e.g., document type, date range)
        vector_weight: Weight for vector search scores (1-vector_weight = keyword weight)

    Returns:
        Tuple of (chunks, scores)
    """
    from .ingest import create_or_load_faiss_index, get_embeddings

    config = get_config()
    if not top_k:
        top_k = config.retrieval.TOP_K

    # Get vector search results first
    vector_ids = []
    try:
        index = create_or_load_faiss_index()
        
        if index.ntotal > 0:
            # Get query embedding
            query_embedding = get_embeddings([query])[0].reshape(1, -1)
            
            # Search FAISS index for top results
            k_search = min(top_k * 10, index.ntotal)  # Get more results than needed for filtering
            distances, indices = index.search(query_embedding, k_search)
            
            # Convert FAISS indices to vector_ids for database lookup
            vector_ids = [int(idx) for idx in indices[0]]
            console.log(f"Vector search found {len(vector_ids)} relevant vectors")
    except Exception as e:
        console.log(f"[bold red]Error in vector search: {str(e)}")
    
    # Get chunks that match metadata filters or vector IDs
    if metadata_filters:
        initial_chunks = get_chunks_by_metadata(metadata_filters, limit=1000)
    else:
        # Use vector_ids to get chunks by similarity
        initial_chunks = get_top_chunks_by_similarity(vector_ids, top_k=1000)

    if not initial_chunks:
        console.log("[bold yellow]No documents match the metadata filters or vector search")
        return [], []

    # Extract text for BM25 search
    corpus = [chunk["text"] for chunk in initial_chunks]
    faiss_ids = [chunk.get("faiss_id") for chunk in initial_chunks]

    # Initialize and fit BM25
    bm25 = BM25(k1=config.bm25.K1, b=config.bm25.B)
    bm25.fit(corpus)

    # Get BM25 scores
    keyword_results = bm25.search(query, top_k=len(corpus))

    # Get vector search scores
    vector_scores = {}
    try:
        # Reuse the previously calculated vector scores
        for faiss_id in vector_ids:
            # Find position in indices array to get corresponding distance
            if faiss_id in faiss_ids:
                idx_pos = vector_ids.index(faiss_id)
                if idx_pos < len(distances[0]):
                    similarity = 1 - (float(distances[0][idx_pos]) ** 2 / 2)
                    vector_scores[int(faiss_id)] = similarity
    except Exception as e:
        console.log(f"[bold red]Error mapping vector scores: {str(e)}")

    # Combine scores
    combined_scores = []
    for i, (doc_idx, keyword_score) in enumerate(keyword_results):
        # Normalize BM25 score (divide by max score if available)
        norm_keyword_score = (
            keyword_score / keyword_results[0][1]
            if keyword_results and keyword_results[0][1] > 0
            else 0
        )

        # Get vector score if available
        faiss_id = faiss_ids[doc_idx]
        vector_score = vector_scores.get(faiss_id, 0.0) if faiss_id is not None else 0.0

        # Weighted combination
        combined_score = (vector_weight * vector_score) + (
            (1 - vector_weight) * norm_keyword_score
        )
        combined_scores.append((doc_idx, combined_score))

    # Sort by combined score
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    combined_scores = combined_scores[:top_k]

    # Get final results
    result_chunks = []
    result_scores = []
    for doc_idx, score in combined_scores:
        if score > 0:
            result_chunks.append(initial_chunks[doc_idx])
            result_scores.append(score)
    
    # Get config
    config = get_config()
    
    # Default number of documents to pass to the LLM
    llm_document_count = 25
    
    # Add reranking step here, but only if enabled
    if config.retrieval.RERANK_ENABLED:
        return rerank_with_cross_encoder(query, result_chunks, result_scores, 
                                         llm_document_count=llm_document_count)
    else:
        # If not reranking, still limit to llm_document_count
        return result_chunks[:llm_document_count], result_scores[:llm_document_count]


def search_documents(
    query: str,
    top_k: Optional[int] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    search_type: str = "hybrid",
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Search for documents relevant to the query.

    Args:
        query: Query string
        top_k: Maximum number of results to return
        metadata_filters: Optional filters for metadata (e.g., document type, date range)
        search_type: Type of search to perform ("vector", "keyword", or "hybrid")

    Returns:
        Tuple of (chunks, scores)
    """
    config = get_config()
    if not top_k:
        top_k = config.retrieval.TOP_K

    # Choose search method based on search_type
    if search_type == "hybrid":
        return hybrid_search(query, top_k, metadata_filters, config.bm25.WEIGHT)
    elif search_type == "keyword":
        # Pure keyword search using BM25
        if metadata_filters:
            chunks = get_chunks_by_metadata(metadata_filters, limit=1000)
        else:
            chunks = get_top_chunks_by_similarity([], top_k=1000)

        if not chunks:
            return [], []

        corpus = [chunk["text"] for chunk in chunks]
        bm25 = BM25(k1=config.bm25.K1, b=config.bm25.B)
        bm25.fit(corpus)

        keyword_results = bm25.search(query, top_k=top_k)

        result_chunks = [chunks[idx] for idx, _ in keyword_results]
        result_scores = [score for _, score in keyword_results]

        # Default number of documents to pass to the LLM
        llm_document_count = 25
        
        # Apply reranking if enabled
        if config.retrieval.RERANK_ENABLED:
            return rerank_with_cross_encoder(query, result_chunks, result_scores, 
                                            llm_document_count=llm_document_count)
        else:
            # If not reranking, still limit to llm_document_count
            return result_chunks[:llm_document_count], result_scores[:llm_document_count]
    else:  # vector search (default fallback)
        from .ingest import create_or_load_faiss_index, get_embeddings

        # Load FAISS index
        index = create_or_load_faiss_index()

        if index.ntotal == 0:
            console.log("[bold red]No documents have been ingested yet!")
            return [], []

        try:
            # Get query embedding
            query_embedding = get_embeddings([query])[0].reshape(1, -1)

            # Search FAISS index
            similarity_threshold = config.retrieval.SCORE_THRESHOLD

            if index.ntotal > 0:
                distances, indices = index.search(
                    query_embedding, min(top_k, index.ntotal)
                )

                # Convert distances to similarity scores
                similarity_scores = [
                    1 - (float(dist) ** 2 / 2) for dist in distances[0]
                ]

                # Filter by similarity threshold
                filtered_indices = [
                    int(idx)
                    for idx, score in zip(indices[0], similarity_scores)
                    if score >= similarity_threshold
                ]
                filtered_scores = [
                    score
                    for score in similarity_scores
                    if score >= similarity_threshold
                ]

                if not filtered_indices:
                    console.log(
                        "[yellow]No results above threshold, returning top results regardless of score"
                    )
                    filtered_indices = indices[0].tolist()
                    filtered_scores = similarity_scores
            else:
                filtered_indices = []
                filtered_scores = []

            # Apply metadata filtering if provided
            chunks = get_top_chunks_by_similarity(filtered_indices, top_k)

            if metadata_filters and chunks:
                filtered_chunks = []
                filtered_chunk_scores = []

                for chunk, score in zip(chunks, filtered_scores):
                    match = True
                    for key, value in metadata_filters.items():
                        if (
                            key == "document_type"
                            and value
                            and chunk.get("document_type") != value
                        ):
                            match = False
                            break
                        elif (
                            key == "project_name"
                            and value
                            and chunk.get("project_name") != value
                        ):
                            match = False
                            break
                        elif (
                            key == "parties_involved"
                            and value
                            and (
                                not chunk.get("parties_involved")
                                or value not in chunk.get("parties_involved")
                            )
                        ):
                            match = False
                            break
                        elif (
                            key == "chunk_type"
                            and value
                            and chunk.get("chunk_type") != value
                        ):
                            match = False
                            break
                        elif (
                            key == "date_from"
                            and value
                            and (
                                not chunk.get("doc_date")
                                or chunk.get("doc_date") < value
                            )
                        ):
                            match = False
                            break
                        elif (
                            key == "date_to"
                            and value
                            and (
                                not chunk.get("doc_date")
                                or chunk.get("doc_date") > value
                            )
                        ):
                            match = False
                            break

                    if match:
                        filtered_chunks.append(chunk)
                        filtered_chunk_scores.append(score)

                return filtered_chunks, filtered_chunk_scores

            # If we have chunks but no scores (edge case), create dummy scores
            if chunks and not filtered_scores:
                filtered_scores = [1.0] * len(chunks)
            
            # Default number of documents to pass to the LLM
            llm_document_count = 25
            
            # Apply reranking if enabled
            if config.retrieval.RERANK_ENABLED and chunks:
                return rerank_with_cross_encoder(query, chunks, filtered_scores, 
                                                llm_document_count=llm_document_count)
            else:
                # If not reranking, still limit to llm_document_count
                return chunks[:llm_document_count], filtered_scores[:llm_document_count]

        except Exception as e:
            console.log(f"[bold red]Error in search: {str(e)}")
            console.log("[bold yellow]Falling back to get all documents")
            return [], []
