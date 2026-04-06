"""
processor.py
------------
Simulates a real-time streaming environment for compliance evaluation.
Accumulates session state across sequential textual chunks, allowing the 
engine to iteratively decide if a violation has occurred in the ongoing 
context. Structurally designed to be easily swappable with LiveKit or
other real WebRTC pipelines in the future.
"""

import logging
from typing import Optional

from compliance_engine.compliance_engine import ComplianceEngine, ComplianceResult

logger = logging.getLogger(__name__)


class StreamingProcessor:
    """
    Stateful streaming wrapper for the Compliance Engine.
    
    Accumulates incoming transcripts to provide the LLM with conversational
    context as to what was said previously.
    """
    
    def __init__(self, engine: ComplianceEngine, domain: str) -> None:
        self.engine = engine
        self.domain = domain
        self._accumulated_context = ""
        self._chunk_count = 0
        self._last_speaker = None
        
    def process_chunk(self, chunk: dict, debug: bool = False) -> Optional[ComplianceResult]:
        """
        Process a new chunk of transcript text incrementally.
        
        Parameters
        ----------
        chunk: A dictionary representing the segmented block (e.g., {"text": "...", "speaker": "agent"}).
        debug: Whether to run the engine in trace mode.
        
        Returns
        -------
        ComplianceResult detailing the current risk state of the entire
        accumulated conversation.
        """
        text = chunk.get("text", "").strip()
        speaker = chunk.get("speaker", "unknown")
        
        if not text:
            return None
            
        if self._last_speaker != speaker:
            formatted_entry = f"\n{speaker.title()}: {text}"
            self._last_speaker = speaker
        else:
            # Continue on the same line if the speaker hasn't changed
            formatted_entry = f" {text}"
            
        if self._accumulated_context:
            self._accumulated_context += formatted_entry
        else:
            self._accumulated_context = formatted_entry.lstrip()
            
        self._chunk_count += 1
        logger.debug("Processing stream chunk %d... length=%d", self._chunk_count, len(self._accumulated_context))
        
        try:
            return self.engine.analyse(
                transcript=self._accumulated_context,
                domain=self.domain,
                debug=debug
            )
        except Exception as exc:
            logger.error("Stream chunk processing failed: %s", exc)
            raise
