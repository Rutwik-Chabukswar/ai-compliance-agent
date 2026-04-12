"""
retriever.py
------------
Policy retrieval using keyword matching only.

FREE MODE (no embeddings):
- Completely removes embedding usage
- No OpenAI API calls
- No vector math or cosine similarity
- Score is the count of query words present in chunk.content
- Preserves class structure and method signatures
"""

import logging
import re
from typing import List, Set, Tuple

from compliance_engine.rag.embeddings import EmbeddingsClient
from compliance_engine.rag.loader import PolicyChunk

logger = logging.getLogger(__name__)


def _extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """
    Extract keywords from text for FREE MODE matching.
    
    Splits text on whitespace/punctuation and returns lowercase words
    with minimum length to filter out noise.
    
    Parameters
    ----------
    text : str
        Text to extract keywords from
    min_length : int
        Minimum word length to include
        
    Returns
    -------
    Set[str]
        Set of extracted keywords
    """
    # Convert to lowercase and split on non-alphanumeric characters
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    # Filter by minimum length and remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    keywords = {w for w in words if len(w) >= min_length and w not in stop_words}
    return keywords


def _keyword_similarity(query: str, chunk_content: str) -> float:
    """
    Compute a simple keyword overlap score.

    FREE MODE scoring:
    - Extract keywords from query and chunk content
    - Count overlapping query keywords present in the chunk
    - Return the raw overlap count as the score
    """
    query_keywords = _extract_keywords(query)
    chunk_keywords = _extract_keywords(chunk_content)
    if not query_keywords:
        return 0.0
    return float(len(query_keywords.intersection(chunk_keywords)))  # FREE MODE


class PolicyRetriever:
    """
    In-memory policy retrieval with FREE MODE support.
    
    FREE MODE (enabled by default if no OpenAI API key):
    - Uses keyword-based matching instead of vector embeddings
    - No external API calls are made
    - Fast, simple, and suitable for development
    
    Normal mode (requires OpenAI API key):
    - Uses vector embeddings and cosine similarity
    - More semantically accurate but requires API quota
    
    Parameters
    ----------
    chunks: List of loaded `PolicyChunk` objects.
    embeddings_client: Optional custom configured `EmbeddingsClient`.
    """

    def __init__(self, chunks: List[PolicyChunk], embeddings_client: EmbeddingsClient | None = None) -> None:
        self.chunks = chunks
        self.client = embeddings_client or EmbeddingsClient()
        self.vectors: List[List[float]] = []

        if self.chunks:
            self._build_index()

    def _build_index(self) -> None:
        """
        FREE MODE index build no-op.

        No embeddings are created or stored in this implementation.
        """
        logger.info("FREE MODE: skipping index build (no embeddings)")
        return

    def retrieve(self, query: str, top_k: int = 3, domain: str | None = None) -> List[Tuple[PolicyChunk, float]]:
        """
        Retrieve the top_k most relevant policy chunks for a query.
        
        In FREE MODE, uses keyword overlap scoring.
        In normal mode, uses vector similarity with cosine distance.
        
        Parameters
        ----------
        query : str
            Query text (typically the transcript)
        top_k : int
            Number of results to return
        domain : str | None
            Optional domain filter (e.g., "fintech", "healthcare")
            
        Returns
        -------
        List[Tuple[PolicyChunk, float]]
            Top k chunks with their similarity scores (descending order)
        """
        if not self.chunks:
            return []

        logger.debug("Retrieving top %d policies (FREE MODE: %s)...", top_k, self.client.use_free_mode)

        scored_chunks: List[Tuple[float, PolicyChunk]] = []

        # FREE MODE: keyword-based matching only
        for chunk in self.chunks:
            if domain and chunk.domain and chunk.domain.lower() != domain.lower():
                continue
            score = _keyword_similarity(query, chunk.content)
            scored_chunks.append((score, chunk))

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        return [(chunk, score) for score, chunk in scored_chunks[:top_k]]
