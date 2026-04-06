"""
retriever.py
------------
In-memory vector store for policy chunks.

Computes embeddings for loaded chunks on startup, and retrieves the closest
K chunks for a given query (transcript) using simple cosine similarity.
"""

import logging
import math
from typing import List, Tuple

from compliance_engine.rag.embeddings import EmbeddingsClient
from compliance_engine.rag.loader import PolicyChunk

logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0


class PolicyRetriever:
    """
    A simple in-memory RAG retriever.
    
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
        """Embed all chunks sequentially."""
        logger.info("Building in-memory vector index for %d policy chunks...", len(self.chunks))
        # For a massive number of chunks you'd want to batch this, but for
        # dozens/hundreds of policy chunks, serial works fine.
        for i, chunk in enumerate(self.chunks):
            vec = self.client.embed_text(chunk.content)
            self.vectors.append(vec)
            if (i + 1) % 10 == 0:
                logger.debug("Embedded %d/%d chunks...", i + 1, len(self.chunks))
        logger.info("Index built successfully.")

    def retrieve(self, query: str, top_k: int = 3, domain: str | None = None) -> List[Tuple[PolicyChunk, float]]:
        """
        Embed the transcript and return the `top_k` most semantically similar
        policy chunks alongside their similarity scores.
        """
        if not self.chunks:
            return []

        logger.debug("Retrieving top %d policies for query...", top_k)
        query_vec = self.client.embed_text(query)

        # Compute scores against all indexed chunks
        scored_chunks: List[Tuple[float, PolicyChunk]] = []
        for vec, chunk in zip(self.vectors, self.chunks):
            if domain and chunk.domain and chunk.domain.lower() != domain.lower():
                continue
            
            sim = cosine_similarity(query_vec, vec)
            scored_chunks.append((sim, chunk))

        # Sort descending by similarity
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        return [(chunk, score) for score, chunk in scored_chunks[:top_k]]
