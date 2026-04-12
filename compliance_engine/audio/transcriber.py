"""
transcriber.py
--------------
Handles audio transcription for the compliance engine in offline FREE MODE.

This module no longer uses external transcription APIs. Audio files are
converted to a local mocked transcript so the pipeline remains functional
without network access.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Service to process audio files and translate them to text offline."""

    def __init__(self, provider: str | None = None) -> None:
        """
        Initialize the transcriber.

        The provider parameter is kept for compatibility, but all offline
        modes use a mocked transcription path.
        """
        self.provider = (provider or "mock").lower()

    def transcribe(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        Convert an audio file to a mocked text transcript.

        Returns:
            Dict containing 'text' (str) and 'confidence' (float).
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info("Transcribing audio file offline: %s via provider=%s", path.name, self.provider)
        return self._mock_transcription(path)

    def _mock_transcription(self, path: Path) -> Dict[str, Any]:
        """Fallback mocked transcription for offline operation."""
        return {
            "text": f"[Transcribed from {path.name}]: We guarantee a 15% return with zero risk!",
            "confidence": 0.5,
        }


def transcribe_audio(audio_path: str | Path, provider: str | None = None) -> Dict[str, Any]:
    """Convenience wrapper for offline audio transcription."""
    return AudioTranscriber(provider=provider).transcribe(audio_path)
