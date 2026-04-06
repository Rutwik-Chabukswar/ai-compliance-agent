"""
loader.py
---------
Loads plain-text policies from a local directory and splits them into smaller chunks.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    """A small segment of a larger policy document."""
    source_file: str
    content: str


def chunk_text(text: str, source_file: str, max_chars: int = 1500, overlap: int = 200) -> List[PolicyChunk]:
    """
    Split text into overlapping chunks of a given max character length.
    Ideally splits on double newlines (paragraphs) to preserve context.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if len(current_chunk) + len(p) + 2 <= max_chars:
            current_chunk += ("\n\n" + p) if current_chunk else p
        else:
            if current_chunk:
                chunks.append(PolicyChunk(source_file, current_chunk))
            
            # If a single paragraph is huge, we just take it as one chunk for now
            # (In a hyper-robust system we'd split it by sentences).
            current_chunk = p

    if current_chunk:
        chunks.append(PolicyChunk(source_file, current_chunk))

    return chunks


def load_policies_from_directory(directory: str | Path) -> List[PolicyChunk]:
    """
    Load all *.txt files from a directory and return them as chunks.
    """
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        logger.warning("Policies directory not found: %s. Returning zero chunks.", directory)
        return []

    all_chunks = []
    
    for file_path in path.glob("**/*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            file_chunks = chunk_text(text, source_file=file_path.name)
            all_chunks.extend(file_chunks)
            logger.debug("Loaded %d chunks from %s", len(file_chunks), file_path.name)
        except Exception as exc:
            logger.error("Failed to read %s: %s", file_path, exc)

    logger.info("Loaded a total of %d policy chunks from %s", len(all_chunks), directory)
    return all_chunks
