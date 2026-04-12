"""
retriever.py
------------
Improved keyword-based policy retrieval system.

Implements normalized keyword overlap scoring with:
- Better text preprocessing
- Normalized scoring (0-1 range)
- Minimum threshold filtering
- Debug information tracking
- Edge case handling
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from compliance_engine.rag.embeddings import EmbeddingsClient
from compliance_engine.rag.loader import PolicyChunk

logger = logging.getLogger(__name__)

# Comprehensive stopword set (common English words that don't carry meaning)
STOPWORDS: Set[str] = {
    # Articles
    "a", "an", "the",
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so",
    # Prepositions
    "in", "on", "at", "to", "for", "from", "with", "by", "about",
    "above", "across", "after", "against", "along", "among", "around",
    "before", "behind", "below", "beneath", "between", "beyond", "down",
    "during", "except", "inside", "into", "like", "near", "of", "off",
    "out", "outside", "over", "through", "throughout", "under", "underneath",
    "until", "upon", "within", "without",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    # Verbs (common)
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "am", "is", "are", "was", "were",
    # Other common words
    "this", "that", "these", "those", "there", "here", "all", "each",
    "every", "both", "either", "neither", "some", "any", "no", "not",
    "as", "if", "what", "which", "who", "when", "where", "why", "how",
}

# Minimum word length to consider (filter out very short tokens)
MIN_WORD_LENGTH: int = 3


@dataclass
class RetrievedChunk:
    """
    Represents a retrieved policy chunk with scoring information.
    
    Attributes
    ----------
    chunk : PolicyChunk
        The original policy chunk from the loader
    score : float
        Normalized similarity score (0.0 to 1.0)
    matched_keywords : List[str]
        Keywords from query that were found in chunk (for debugging)
    query_keyword_count : int
        Total number of keywords in query (for debugging)
    """
    
    chunk: PolicyChunk
    score: float
    matched_keywords: List[str]
    query_keyword_count: int

    def __repr__(self) -> str:
        """Pretty representation for logging/debugging."""
        source = self.chunk.source_file
        domain = f" ({self.chunk.domain})" if self.chunk.domain else ""
        matches = len(self.matched_keywords)
        return (
            f"RetrievedChunk(source={source}{domain}, score={self.score:.2f}, "
            f"matches={matches}/{self.query_keyword_count})"
        )


def preprocess_text(text: str) -> str:
    """
    Preprocess text for keyword extraction.
    
    Operations:
    1. Convert to lowercase
    2. Normalize whitespace (collapse multiple spaces)
    3. Keep alphanumeric and spaces/punctuation for later splitting
    
    Parameters
    ----------
    text : str
        Raw text to preprocess
        
    Returns
    -------
    str
        Preprocessed text (lowercase, normalized spaces)
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_keywords(text: str, min_length: int = MIN_WORD_LENGTH) -> Tuple[Set[str], List[str]]:
    """
    Extract meaningful keywords from text.
    
    Process:
    1. Preprocess text (lowercase, normalize spaces)
    2. Split on non-alphanumeric characters
    3. Remove stopwords
    4. Filter by minimum length
    5. Remove duplicates
    
    Parameters
    ----------
    text : str
        Text to extract keywords from
    min_length : int
        Minimum word length to include (default 3)
        
    Returns
    -------
    Tuple[Set[str], List[str]]
        - Set of unique keywords (for set operations)
        - List of keywords in order (for debugging)
    """
    if not text:
        return set(), []
    
    # Preprocess
    text = preprocess_text(text)
    
    # Split on non-alphanumeric characters and filter
    words = re.findall(r'\b[a-z0-9]+\b', text)
    
    # Filter: minimum length, not stopword, not number-only
    keywords_list: List[str] = []
    for word in words:
        if (len(word) >= min_length and 
            word not in STOPWORDS and 
            not word.isdigit()):
            keywords_list.append(word)
    
    # Convert to set for efficient intersection operations
    keywords_set = set(keywords_list)
    
    logger.debug(
        "Extracted %d keywords from text (length=%d): %s",
        len(keywords_set),
        len(text),
        list(keywords_set)[:5] + (["..."] if len(keywords_set) > 5 else [])
    )
    
    return keywords_set, keywords_list


def compute_similarity_score(
    query_keywords: Set[str],
    chunk_keywords: Set[str],
    query_keyword_count: int,
) -> Tuple[float, List[str]]:
    """
    Compute normalized similarity score between query and chunk.
    
    Scoring Formula:
        score = overlap_keywords / total_query_keywords
    
    Where:
    - overlap_keywords = len(query_keywords ∩ chunk_keywords)
    - total_query_keywords = number of keywords in query
    
    This ensures score is in range [0.0, 1.0].
    
    Parameters
    ----------
    query_keywords : Set[str]
        Keywords extracted from query
    chunk_keywords : Set[str]
        Keywords extracted from chunk
    query_keyword_count : int
        Total number of keywords in query (for normalization)
        
    Returns
    -------
    Tuple[float, List[str]]
        - Normalized score (0.0 to 1.0)
        - List of matched keywords (intersection)
    """
    if query_keyword_count == 0:
        return 0.0, []
    
    # Find keywords that appear in both query and chunk
    matched_keywords = list(query_keywords.intersection(chunk_keywords))
    
    # Normalize by total query keywords
    score = len(matched_keywords) / query_keyword_count
    
    return score, matched_keywords


def filter_and_rank(
    scored_results: List[Tuple[PolicyChunk, float, List[str], int]],
    top_k: int = 3,
    min_score: float = 0.2,
) -> List[RetrievedChunk]:
    """
    Filter by minimum score threshold and rank by score.
    
    Parameters
    ----------
    scored_results : List[Tuple[PolicyChunk, float, List[str], int]]
        List of (chunk, score, matched_keywords, query_keyword_count) tuples
    top_k : int
        Maximum number of results to return (default 3)
    min_score : float
        Minimum score threshold to include result (default 0.2)
        
    Returns
    -------
    List[RetrievedChunk]
        Top-k results filtered and ranked by score (descending)
    """
    # Filter by minimum score
    filtered: List[RetrievedChunk] = []
    for chunk, score, matched_keywords, query_keyword_count in scored_results:
        if score >= min_score:
            filtered.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                matched_keywords=matched_keywords,
                query_keyword_count=query_keyword_count,
            ))
    
    # Sort by score descending
    filtered.sort(key=lambda x: x.score, reverse=True)
    
    # Return top k
    results = filtered[:top_k]
    
    logger.debug(
        "After filtering (threshold=%.2f) and ranking: %d results (of %d scored)",
        min_score,
        len(results),
        len(scored_results),
    )
    
    return results


class PolicyRetriever:
    """
    Improved in-memory policy retrieval using keyword-based matching.
    
    Features:
    - Normalized similarity scoring (0.0 to 1.0 range)
    - Minimum score threshold filtering
    - Debug information (matched keywords)
    - Domain filtering
    - Edge case handling
    
    Parameters
    ----------
    chunks : List[PolicyChunk]
        List of loaded policy chunks
    embeddings_client : Optional[EmbeddingsClient]
        (Unused, kept for compatibility)
    min_score_threshold : float
        Minimum score to include results (default 0.2)
    min_word_length : int
        Minimum word length for keywords (default 3)
    """

    def __init__(
        self,
        chunks: List[PolicyChunk],
        embeddings_client: Optional[EmbeddingsClient] = None,
        min_score_threshold: float = 0.2,
        min_word_length: int = MIN_WORD_LENGTH,
    ) -> None:
        self.chunks = chunks or []
        self.client = embeddings_client or EmbeddingsClient()
        self.min_score_threshold = min_score_threshold
        self.min_word_length = min_word_length
        self.vectors: List[List[float]] = []
        
        if self.chunks:
            self._build_index()
        
        logger.info(
            "PolicyRetriever initialized (chunks=%d, min_score=%.2f, "
            "min_word_length=%d)",
            len(self.chunks),
            self.min_score_threshold,
            self.min_word_length,
        )

    def _build_index(self) -> None:
        """
        Build keyword index (no-op in keyword-based mode).
        
        Kept for API compatibility with potential future vector-based implementations.
        """
        logger.info("Keyword-based retriever: skipping embedding index build")
        return

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        domain: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most relevant policy chunks for a query.
        
        Process:
        1. Extract keywords from query
        2. Score all chunks using normalized similarity
        3. Filter by domain (if specified)
        4. Apply minimum score threshold
        5. Return top_k ranked by score
        
        Parameters
        ----------
        query : str
            Query text (typically the transcript or excerpt)
        top_k : int
            Number of results to return (default 3)
        domain : Optional[str]
            Optional domain filter (e.g., "fintech", "healthcare", "insurance")
            
        Returns
        -------
        List[RetrievedChunk]
            Top-k retrieved chunks with scores and debug info, sorted by score (descending)
            Empty list if no matches or invalid query.
        """
        # Edge case: empty or invalid source
        if not self.chunks:
            logger.warning("PolicyRetriever.retrieve() called with no chunks loaded")
            return []
        
        # Edge case: empty query
        if not query or not query.strip():
            logger.warning("PolicyRetriever.retrieve() called with empty query")
            return []
        
        # Extract query keywords
        query_keyword_set, query_keyword_list = extract_keywords(
            query,
            min_length=self.min_word_length
        )
        
        # Edge case: query has no meaningful keywords
        if not query_keyword_set:
            logger.warning(
                "Query '%s' has no meaningful keywords after preprocessing",
                query[:50]
            )
            return []
        
        logger.debug(
            "Retrieving policies for query (length=%d, keywords=%d, domain=%s, top_k=%d)",
            len(query),
            len(query_keyword_set),
            domain or "any",
            top_k,
        )
        
        scored_chunks: List[Tuple[PolicyChunk, float, List[str], int]] = []
        
        # Score all chunks
        for chunk in self.chunks:
            # Apply domain filter if specified
            if domain and chunk.domain and chunk.domain.lower() != domain.lower():
                continue
            
            # Extract chunk keywords
            chunk_keyword_set, _ = extract_keywords(
                chunk.content,
                min_length=self.min_word_length
            )
            
            # Compute normalized score
            score, matched_keywords = compute_similarity_score(
                query_keyword_set,
                chunk_keyword_set,
                len(query_keyword_set),
            )
            
            # Store result
            scored_chunks.append((chunk, score, matched_keywords, len(query_keyword_set)))
        
        logger.debug(
            "Scored %d chunks (after domain filter: %s)",
            len(scored_chunks),
            domain or "none"
        )
        
        # Filter and rank
        results = filter_and_rank(
            scored_chunks,
            top_k=top_k,
            min_score=self.min_score_threshold,
        )
        
        # Log results
        if results:
            logger.debug(
                "Retrieved %d policy chunks: %s",
                len(results),
                " | ".join(str(r) for r in results)
            )
        else:
            logger.debug(
                "No policy chunks retrieved "
                "(min_score_threshold=%.2f, scored=%d, domain=%s)",
                self.min_score_threshold,
                len(scored_chunks),
                domain or "any",
            )
        
        return results

