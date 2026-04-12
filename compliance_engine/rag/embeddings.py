"""
embeddings.py
-------------
FREE MODE Embeddings client.

FREE MODE (no embeddings):
- Returns a dummy vector for all text inputs
- Eliminates any dependency on OpenAI or external embedding APIs
- Ensures the system runs without API quota or billing
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """
    Dummy embeddings client for FREE MODE.
    
    This class preserves the original API surface while ensuring that no
    OpenAI calls are ever made.
    """

    def __init__(
        self,
        model: str = "dummy",
        api_key: str | None = None,
        base_url: str | None = None,
        use_free_mode: bool | None = None,
    ) -> None:
        """
        Initialize the dummy embeddings client.

        Parameters
        ----------
        model : str
            Ignored in FREE MODE.
        api_key : str | None
            Ignored in FREE MODE.
        base_url : str | None
            Ignored in FREE MODE.
        use_free_mode : bool | None
            Ignored; FREE MODE is always enforced.
        """
        self.use_free_mode = True
        logger.info("Embeddings client initialized in FREE MODE (no API calls)")

    def embed_text(self, text: str) -> List[float]:
        """
        Return a dummy vector for the given text.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        List[float]
            Dummy vector [0.0].
        """
        return [0.0]  # FREE MODE

