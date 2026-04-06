"""
embeddings.py
-------------
Wrapper for calling the OpenAI Embeddings API.
Converts text strings or policy chunks into vector representations.
"""

import logging
from typing import List

from openai import OpenAI

from compliance_engine.config import DEFAULT_EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Simple wrapper for OpenAI embeddings."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_text(self, text: str) -> List[float]:
        """Convert a single string into a dense vector."""
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        
        logger.debug("Generating embedding for text (len=%d) using %s", len(text), self.model)
        
        # Replace newlines (recommended by OpenAI)
        text = text.replace("\n", " ")
        
        response = self._client.embeddings.create(
            input=[text],
            model=self.model,
        )
        return response.data[0].embedding
