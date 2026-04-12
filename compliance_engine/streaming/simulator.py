"""
simulator.py
------------
Simulation layer for real-time streaming compliance detection.
"""

import logging
import time
from typing import List, Dict, Any, Generator

from compliance_engine.agent import ComplianceAgent

logger = logging.getLogger(__name__)


def chunk_transcript(text: str, chunk_size: int = 15) -> List[str]:
    """
    Splits a full transcript into smaller chunks by word count
    to simulate real-time streaming segments.
    
    Parameters
    ----------
    text : str
        The full transcript text.
    chunk_size : int, optional
        The number of words per chunk, by default 15.
        
    Returns
    -------
    List[str]
        A list of string chunks.
    """
    if not text:
        return []
        
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


class StreamingComplianceProcessor:
    """
    Simulates real-time streaming compliance detection by processing
    a transcript sequentially in chunks.
    """
    
    def __init__(self, agent: ComplianceAgent, chunk_size: int = 15):
        """
        Initialize the streaming processor with the core compliance agent.
        
        Parameters
        ----------
        agent : ComplianceAgent
            The underlying decision system used for evaluating chunks.
        chunk_size : int, optional
            Number of words per simulated chunk, by default 15
        """
        self.agent = agent
        self.chunk_size = chunk_size
        self.accumulated_context: List[str] = []

    def process_stream(self, transcript: str, domain: str = "general") -> Generator[Dict[str, Any], None, None]:
        """
        Processes a full transcript by chunking it and evaluating sequentially.
        Maintains state across chunks and yields an early alert if a violation is detected.
        
        Parameters
        ----------
        transcript : str
            The full transcript to process.
        domain : str, optional
            The domain context passed to the agent, by default "general".
            
        Yields
        ------
        Dict[str, Any]
            A dictionary containing the following keys:
            - chunk: str
            - violation: bool
            - suggestion: str
        """
        self.accumulated_context = []
        chunks = chunk_transcript(transcript, self.chunk_size)

        logger.info("Starting stream simulation with %d chunks", len(chunks))

        for i, chunk in enumerate(chunks):
            start_time = time.time()
            
            # Maintain stateful background context from previous chunks
            self.accumulated_context.append(chunk)
            context_text = " ".join(self.accumulated_context)
            
            logger.info("Processing chunk %d/%d: '%s'", i + 1, len(chunks), chunk)
            
            # Pass accumulated context to agent to make informed compliance decision
            result = self.agent.analyse(transcript=context_text, domain=domain)
            
            process_time = time.time() - start_time
            logger.debug("Chunk %d detection timing: %.3f seconds", i + 1, process_time)
            
            output = {
                "chunk": chunk,
                "violation": result.violation,
                "suggestion": result.suggestion
            }
            
            yield output
            
            # Early Detection: Yield alert and stop if a violation is found
            if result.violation:
                logger.warning("Violation detected early at chunk %d. Halting stream simulation.", i + 1)
                break
