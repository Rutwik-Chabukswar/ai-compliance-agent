"""
transcriber.py
--------------
Handles audio transcription for the compliance engine.
Supports Sarvam AI and OpenAI Whisper with automatic fallback to mock parsing.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Service to process audio files and translate them to text."""
    
    def __init__(self, provider: str | None = None) -> None:
        """
        Initialize the transcriber.
        :param provider: "whisper" or "sarvam". If None, defaults to exploring env vars.
        """
        self.provider = provider or os.getenv("COMPLIANCE_AUDIO_PROVIDER", "whisper").lower()
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.sarvam_key = os.getenv("SARVAM_API_KEY", "")
        
    def transcribe(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        Convert an audio file to a text transcript using the selected provider.
        
        Returns:
            Dict containing 'text' (str) and 'confidence' (float, optional).
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
            
        print(f"[•] Using Audio Provider: {self.provider.upper()}")
        logger.info("Transcribing audio file: %s via %s", path.name, self.provider)
        
        try:
            if self.provider == "whisper":
                return self._transcribe_whisper(path)
            elif self.provider == "sarvam":
                return self._transcribe_sarvam(path)
            else:
                logger.warning("Unknown provider '%s'. Falling back to mock.", self.provider)
                return self._mock_transcription(path)
        except Exception as exc:
            logger.warning("Provider '%s' failed: %s. Falling back to mock.", self.provider, exc)
            return self._mock_transcription(path)

    def _transcribe_whisper(self, path: Path) -> Dict[str, Any]:
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
            
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        with open(path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="verbose_json"
            )
            
        # verbose_json returns segments with confidence data (OpenAI doesn't always populate top-level confidence reliably on some endpoints, but we attempt)
        confidence = None
        if hasattr(transcript, "segments") and transcript.segments:
            # Average confidence of segments
            confs = [s["confidence"] for s in getattr(transcript, "segments", []) if "confidence" in s]
            if confs:
                confidence = sum(confs) / len(confs)
                
        return {
            "text": str(transcript.text),
            "confidence": confidence
        }

    def _transcribe_sarvam(self, path: Path) -> Dict[str, Any]:
        if not self.sarvam_key:
            raise ValueError("SARVAM_API_KEY not found in environment.")
            
        # Minimal implementation using requests for Sarvam AI spec
        import requests
        
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {
            "api-subscription-key": self.sarvam_key
        }
        
        with open(path, "rb") as audio_file:
            files = {"file": (path.name, audio_file, "audio/wav")}
            # model could be saaras depending on latest Sarvam endpoint, passing arbitrary parameters for example setup
            data = {"model": "saaras:v1"}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            
        response.raise_for_status()
        result = response.json()
        
        # Adjust parsing according to Sarvam's exact JSON API format mapping
        return {
            "text": result.get("transcript", ""),
            "confidence": result.get("confidence", None)
        }

    def _mock_transcription(self, path: Path) -> Dict[str, Any]:
        """Fallback mocked extraction."""
        return {
            "text": f"[Transcribed from {path.name}]: We guarantee a 15% return with zero risk!",
            "confidence": 0.5
        }


def transcribe_audio(audio_path: str | Path, provider: str | None = None) -> Dict[str, Any]:
    """Convenience functional wrapper for the default transcriber."""
    return AudioTranscriber(provider=provider).transcribe(audio_path)
