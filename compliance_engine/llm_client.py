"""
llm_client.py
-------------
Production-ready LLM client with hybrid mode support.

This module provides:
  - Real LLM-based reasoning when API key is available
  - Reliable fallback to rule-based detection when LLM is unavailable
  - Retry logic with exponential backoff
  - Structured JSON response enforcement
  - Comprehensive error handling and logging

The client maintains backward compatibility with the ComplianceEngine
interface while providing a clean, production-grade implementation.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from compliance_engine.config import (
    DEFAULT_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MAX_RETRIES,
    MAX_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_FACTOR,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Try to import openai client; graceful fallback if not installed
try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai library not installed; will use rule-based fallback exclusively")


@dataclass
class LLMResponse:
    """Structured compliance analysis result from LLM or fallback engine."""

    violation: bool
    confidence: float
    reason: str
    debug: Optional[Dict[str, Any]] = None


class LLMClientError(Exception):
    """Raised when LLMClient cannot produce a valid response."""


class LLMClient:
    """
    Hybrid LLM client with fallback rule-based detection.

    Attempts to use OpenAI API when credentials are available. Falls back
    gracefully to rule-based detection if the LLM is unavailable, fails,
    or returns malformed responses.

    Parameters
    ----------
    model : str
        OpenAI model name (e.g., "gpt-4o-mini"). Default from config.
    api_key : str, optional
        OpenAI API key. Default from OPENAI_API_KEY environment variable.
    base_url : str, optional
        OpenAI API base URL. Default from OPENAI_BASE_URL environment variable.
    temperature : float
        Sampling temperature for LLM. Default 0.0 for determinism.
    max_tokens : int
        Maximum tokens in response. Default 512.
    max_retries : int
        Maximum retry attempts for transient failures. Default 3.
    timeout : int
        Request timeout in seconds. Default 30.
    retry_backoff : float
        Exponential backoff multiplier between retries. Default 1.5.
    use_fallback_on_error : bool
        If True, silently fall back to rule-based engine on LLM failures.
        If False, raise exception. Default True.
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
        use_fallback_on_error: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_backoff = retry_backoff
        self.use_fallback_on_error = use_fallback_on_error

        # Detect API availability
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or OPENAI_BASE_URL
        self.llm_available = HAS_OPENAI and bool(self.api_key)

        # Initialize OpenAI client if credentials are available
        self.client: Optional[OpenAI] = None
        if self.llm_available:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                    timeout=self.timeout,
                )
                logger.info("LLMClient initialized in OPENAI mode")
            except Exception as e:
                logger.warning(
                    "Failed to initialize OpenAI client: %s. Falling back to "
                    "rule-based mode.",
                    str(e),
                )
                self.llm_available = False
                self.client = None
        else:
            logger.info("LLMClient using fallback mode")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        system_context: Optional[str] = None,
    ) -> str:
        """
        Main entry point: performs LLM-based or rule-based compliance analysis.

        Attempts to call the LLM with the provided prompts. If the LLM is
        unavailable or fails, falls back to rule-based detection.

        Parameters
        ----------
        system_prompt : str
            System-level instructions for the model (compliance rules, context).
        user_prompt : str
            The transcript or text to analyze.
        system_context : str, optional
            Additional policy or RAG context to prepend to the system prompt.

        Returns
        -------
        str
            JSON-encoded compliance decision with structure:
            {
                "violation": bool,
                "risk_level": "low" | "medium" | "high",
                "confidence": float,
                "reason": str,
                "suggestion": str
            }

        Raises
        ------
        LLMClientError
            Only if use_fallback_on_error is False and all attempts fail.
        """
        # Attempt LLM-based analysis first
        if self.llm_available and self.client is not None:
            try:
                full_prompt = system_prompt
                if system_context:
                    full_prompt = f"{system_prompt}\n\n{system_context}"
                full_prompt = f"{full_prompt}\n\nTranscript:\n{user_prompt}"

                response_text = self._call_llm(
                    prompt=full_prompt
                )
                parsed = self._parse_response(response_text)
                logger.debug("LLM analysis succeeded for transcript length=%d", len(user_prompt))
                return json.dumps(parsed)
            except Exception as e:
                logger.warning(
                    "LLM analysis failed (%s: %s). Falling back to rule-based detection.",
                    type(e).__name__,
                    str(e),
                )
                if not self.use_fallback_on_error:
                    raise LLMClientError(f"LLM call failed and fallback disabled: {str(e)}") from e

        # Fall back to rule-based detection
        logger.debug("Using rule-based fallback for transcript length=%d", len(user_prompt))
        fallback_result = self._fallback_rule_engine(user_prompt)
        return json.dumps(fallback_result)

    def analyse(
        self,
        transcript: str,
        context: Optional[str] = None,
        debug: bool = False,
    ) -> LLMResponse:
        """
        Backward-compatible wrapper for the old LLM API.

        Delegates to chat() but returns an LLMResponse dataclass for
        internal compatibility with ComplianceEngine.

        Parameters
        ----------
        transcript : str
            The transcript to analyze.
        context : str, optional
            Additional context (usually RAG-provided policy text).
        debug : bool
            If True, include debug metadata in response.

        Returns
        -------
        LLMResponse
            Compliance analysis result as typed dataclass.
        """
        # Build minimal system prompt for rule-based fallback
        system_prompt = (
            "You are a compliance officer. Analyze the transcript for violations."
        )
        if context:
            system_prompt = f"{system_prompt}\n\nPolicy Context:\n{context}"

        json_result = self.chat(
            system_prompt=system_prompt,
            user_prompt=transcript,
            system_context=None,
        )

        try:
            data = json.loads(json_result)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse chat response as JSON: %s", str(e))
            raise LLMClientError(f"Invalid JSON response: {str(e)}") from e

        debug_payload = None
        if debug:
            debug_payload = {
                "matched_rule": data.get("matched_rule", "UNKNOWN"),
                "reasoning_summary": data.get("reason", ""),
            }

        return LLMResponse(
            violation=data.get("violation", False),
            confidence=data.get("confidence", 0.0),
            reason=data.get("reason", ""),
            debug=debug_payload,
        )

    # =========================================================================
    # Private: LLM Integration
    # =========================================================================

    def _call_llm(self, prompt: str) -> str:
        """
        Call OpenAI API with retry logic and exponential backoff.

        Attempts up to max_retries times, backing off exponentially
        between attempts. Handles transient network errors and rate limits.

        Parameters
        ----------
        prompt : str
            Combined prompt containing context and transcript.

        Returns
        -------
        str
            Raw response text from the model.

        Raises
        ------
        LLMClientError
            If all retry attempts fail or response is invalid.
        """
        if not self.client:
            raise LLMClientError("OpenAI client is not initialized")

        attempt = 0
        last_error = None
        backoff_time = 0.5  # Start with 500ms

        while attempt < self.max_retries:
            try:
                logger.debug(
                    "LLM API call attempt %d/%d (model=%s, timeout=%ds)",
                    attempt + 1,
                    self.max_retries,
                    self.model,
                    self.timeout,
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a strict compliance officer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                # Extract text from response
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content or ""

                raise LLMClientError("Empty response from LLM")

            except (APIConnectionError, APIError) as e:
                # Transient network or API errors - retry
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(
                        "Transient API error (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt,
                        self.max_retries,
                        type(e).__name__,
                        backoff_time,
                    )
                    time.sleep(backoff_time)
                    backoff_time *= self.retry_backoff
                continue

            except RateLimitError as e:
                # Rate limited - retry with longer backoff
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(
                        "Rate limited (attempt %d/%d). Retrying in %.1fs...",
                        attempt,
                        self.max_retries,
                        backoff_time * 2,
                    )
                    time.sleep(backoff_time * 2)
                    backoff_time *= self.retry_backoff
                continue

            except Exception as e:
                # Unknown error - don't retry
                logger.error("Unexpected error in LLM call: %s", str(e))
                raise LLMClientError(f"LLM call failed: {str(e)}") from e

        # All retries exhausted
        raise LLMClientError(
            f"LLM call failed after {self.max_retries} attempts: {str(last_error)}"
        )

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response as structured JSON.

        Extracts JSON from the response (with or without markdown
        formatting), validates the schema, and handles edge cases.

        Parameters
        ----------
        response_text : str
            Raw response text from LLM model.

        Returns
        -------
        dict
            Validated compliance decision with structure:
            {
                "violation": bool,
                "risk_level": str,
                "confidence": float,
                "reason": str,
                "suggestion": str
            }

        Raises
        ------
        LLMClientError
            If JSON is invalid or schema validation fails.
        """
        # Try to extract JSON from response (handle markdown fences, etc.)
        json_text = self._extract_json_block(response_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", str(e))
            raise LLMClientError(f"Invalid JSON in response: {str(e)}") from e

        # Validate required fields
        required_fields = {"violation", "risk_level", "confidence", "reason", "suggestion"}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise LLMClientError(
                f"Response missing required fields: {', '.join(missing_fields)}"
            )

        # Validate and coerce types
        try:
            # Ensure violation is boolean
            if isinstance(data["violation"], str):
                data["violation"] = data["violation"].lower() in {"true", "1", "yes"}
            elif not isinstance(data["violation"], bool):
                data["violation"] = bool(data["violation"])

            # Ensure risk_level is valid
            valid_risk_levels = {"low", "medium", "high"}
            if data["risk_level"].lower() not in valid_risk_levels:
                raise ValueError(
                    f"risk_level must be one of {valid_risk_levels}, "
                    f"got {data['risk_level']}"
                )
            data["risk_level"] = data["risk_level"].lower()

            # Ensure confidence is float in [0, 1]
            confidence = float(data["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be in [0, 1], got {confidence}")
            data["confidence"] = confidence

            # Ensure reason and suggestion are strings
            data["reason"] = str(data.get("reason", "")).strip()
            data["suggestion"] = str(data.get("suggestion", "")).strip()

        except (ValueError, AttributeError, TypeError) as e:
            raise LLMClientError(f"Response validation failed: {str(e)}") from e

        logger.debug(
            "LLM response parsed successfully: violation=%s, risk=%s, confidence=%.2f",
            data["violation"],
            data["risk_level"],
            data["confidence"],
        )

        return data

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Extract JSON object from text (handles markdown fences, preamble).

        Parameters
        ----------
        text : str
            Text potentially containing a JSON object.

        Returns
        -------
        str
            Extracted JSON string.

        Raises
        ------
        LLMClientError
            If no valid JSON is found or JSON is malformed.
        """
        # Remove markdown code fences if present
        text = re.sub(r"```(?:json)?\n?", "", text)

        # Find the first '{' and the last '}'
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            raise LLMClientError(f"No JSON object found in response: {text[:100]}")

        json_str = text[start_idx : end_idx + 1]

        # Validate JSON is balanced
        if json_str.count("{") != json_str.count("}"):
            raise LLMClientError(f"Unbalanced braces in JSON: {json_str[:100]}")

        return json_str

    # =========================================================================
    # Private: Rule-Based Fallback
    # =========================================================================

    def _fallback_rule_engine(self, transcript: str) -> Dict[str, Any]:
        """
        Rule-based compliance detection for offline/fallback scenarios.

        Detects high-risk keywords and phrases that commonly indicate
        compliance violations in fintech/insurance domains.

        Parameters
        ----------
        transcript : str
            The transcript to analyze.

        Returns
        -------
        dict
            Compliance decision (same schema as LLM output).
        """
        text = transcript.lower()

        # Define risk thresholds and patterns
        high_risk_phrases = [
            "guaranteed return",
            "guaranteed returns",
            "no risk",
            "zero risk",
            "risk free",
            "risk-free",
            "assured profit",
            "cannot lose",
            "will not lose",
        ]

        medium_risk_phrases = [
            "high returns",
            "limited time offer",
            "exclusive deal",
            "best opportunity",
            "don't miss out",
            "act now",
        ]

        # Check for high-risk patterns
        if any(phrase in text for phrase in high_risk_phrases) or (
            "guarantee" in text and ("return" in text or "profit" in text)
        ):
            violation = True
            risk_level = "high"
            confidence = 0.95
            reason = "Detected guaranteed return or zero-risk claim, which is prohibited."
            suggestion = (
                "Remove guaranteed return language and add appropriate "
                "risk disclosures."
            )
        # Check for medium-risk patterns
        elif any(phrase in text for phrase in medium_risk_phrases):
            violation = True
            risk_level = "medium"
            confidence = 0.70
            reason = "Detected potentially misleading promotional language."
            suggestion = (
                "Add appropriate disclaimers and balance promotional claims "
                "with risk information."
            )
        # Compliant
        else:
            violation = False
            risk_level = "low"
            confidence = 0.85
            reason = "No explicit high-risk claims detected in transcript."
            suggestion = "No action required."

        logger.debug(
            "Rule-based fallback analysis: violation=%s, risk=%s, confidence=%.2f",
            violation,
            risk_level,
            confidence,
        )

        return {
            "violation": violation,
            "risk_level": risk_level,
            "confidence": confidence,
            "reason": reason,
            "suggestion": suggestion,
        }
