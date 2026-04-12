"""
llm_client.py
-------------
Local rule-based compliance engine for fully offline FREE MODE.

This module preserves the existing LLMClient interface while replacing
external API calls with a simple, local phrase matcher.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from compliance_engine.config import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    MAX_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_FACTOR,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Local rule-based compliance analysis result."""

    violation: bool
    confidence: float
    reason: str
    debug: Optional[dict[str, Any]] = None


class LLMClientError(Exception):
    """Raised when the local LLM client cannot produce a response."""


class LLMClient:
    """
    Offline compliance client that performs local rule-based analysis.

    Parameters
    ----------
    model:
        Placeholder model identifier for compatibility.
    api_key:
        Ignored in offline mode.
    base_url:
        Ignored in offline mode.
    temperature:
        Ignored in offline mode.
    max_tokens:
        Ignored in offline mode.
    max_retries:
        Ignored in offline mode.
    timeout:
        Ignored in offline mode.
    retry_backoff:
        Ignored in offline mode.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = MAX_RETRIES,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        retry_backoff: float = RETRY_BACKOFF_FACTOR,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_backoff = retry_backoff

        logger.info("LLMClient initialized in FULLY OFFLINE FREE MODE")

    def analyse(
        self,
        transcript: str,
        context: Optional[str] = None,
        debug: bool = False,
    ) -> LLMResponse:
        """
        Perform rule-based compliance analysis locally.

        Parameters
        ----------
        transcript:
            The transcript text to evaluate.
        context:
            Optional policy context that may increase confidence.
        debug:
            When ``True``, include local reasoning metadata for debugging.

        Returns
        -------
        LLMResponse
            Local analysis result.
        """
        text = transcript.lower()

        high_risk_phrases = [
            "guaranteed return",
            "no risk",
            "zero risk",
            "risk free",
            "assured profit",
        ]
        medium_risk_phrases = [
            "high returns",
            "limited time offer",
            "exclusive deal",
        ]

        if any(phrase in text for phrase in high_risk_phrases) or (
            "guarantee" in text and "return" in text
        ):
            violation = True
            confidence = 0.9
            reason = "Detected a high-risk financial claim."
            matched_rule = "MISLEADING_GUARANTEE"
            reasoning_summary = "The transcript contains a prohibited guaranteed return claim."
        elif any(phrase in text for phrase in medium_risk_phrases):
            violation = True
            confidence = 0.6
            reason = "Detected a medium-risk promotional claim."
            matched_rule = "MISLEADING_GUARANTEE"
            reasoning_summary = "The transcript contains a potentially misleading promotional claim."
        else:
            violation = False
            confidence = 0.1
            reason = "No explicit high-risk or medium-risk claim detected."
            matched_rule = "COMPLIANT"
            reasoning_summary = "No disallowed claims were identified in the transcript."

        if context:
            query_words = set(re.findall(r"\b[a-z0-9]+\b", text))
            context_words = set(re.findall(r"\b[a-z0-9]+\b", context.lower()))
            if query_words and context_words:
                overlap = len(query_words.intersection(context_words))
                if overlap > 0:
                    confidence = min(1.0, confidence + 0.1)
                    reason += " Matching policy context increased confidence."

        debug_payload: Optional[dict[str, Any]] = None
        if debug:
            debug_payload = {
                "matched_rule": matched_rule,
                "reasoning_summary": reasoning_summary,
            }

        return LLMResponse(
            violation=violation,
            confidence=confidence,
            reason=reason,
            debug=debug_payload,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        system_context: Optional[str] = None,
    ) -> str:
        """
        Compatibility wrapper that returns JSON output.

        Existing callers may still invoke `.chat()`, so this maintains the
        legacy interface while delegating to offline analysis.
        """
        result = self.analyse(transcript=user_prompt, context=system_context)
        return json.dumps(
            {
                "violation": result.violation,
                "confidence": result.confidence,
                "reason": result.reason,
            }
        )
