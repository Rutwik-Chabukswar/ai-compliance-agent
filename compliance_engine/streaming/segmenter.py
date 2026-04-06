"""
segmenter.py
------------
Simulates an advanced voice activity detection (VAD) and diarization pipeline
(inspired by Celerio or external real-time diarization APIs).

Takes a raw transcript block, parses out speaker identities, and segments 
text dynamically across acoustic/punctuation boundaries and pauses.
"""

import re
from typing import Any, Dict, List


def simulate_segmentation(full_text: str) -> List[Dict[str, str]]:
    """
    Parses a combined transcript and simulates intelligent chunking:
    - Extracts "Agent:" or "Customer:" labels via regex.
    - Yields chunks broken across sentence boundaries, punctuation, or 
      simulated pauses (e.g. commas).
      
    Returns a list of structured dictionaries ready for streaming logic.
    """
    results: List[Dict[str, str]] = []
    
    # Capture known speakers to simulate diarization
    parts = re.split(r'\b(Agent|Customer):\s*', full_text, flags=re.IGNORECASE)
    
    current_speaker = "unknown"
    blocks: List[tuple[str, str]] = []
    
    if len(parts) == 1:
        blocks = [(current_speaker, parts[0])]
    else:
        # parts[0] is typically empty if the string starts with "Agent:"
        if parts[0].strip():
            blocks.append((current_speaker, parts[0]))
            
        for i in range(1, len(parts), 2):
            speaker = parts[i].strip().lower()
            text = parts[i+1]
            blocks.append((speaker, text))
            
    # Segment each unified block into smaller 'breath/pause' chunks
    for speaker, text_block in blocks:
        # Split on sentence boundaries (.!?) or pauses (,) but KEEP the punctuation attached
        # We split spaces AFTER punctuation marks
        raw_chunks = re.split(r'(?<=[.!?,-])\s+', text_block)
        
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if chunk:
                results.append({
                    "speaker": speaker,
                    "text": chunk
                })
                
    return results
