"""
Audio module initialization.
"""
from compliance_engine.audio.transcriber import AudioTranscriber, transcribe_audio

__all__ = [
    "AudioTranscriber",
    "transcribe_audio",
]
