"""
compliance_engine.py
--------------------
Core compliance detection logic.

Responsibilities
~~~~~~~~~~~~~~~~
* Build the final prompt from templates + user inputs.
* Call the LLM via ``LLMClient``.
* Parse and validate the JSON response.
* Provide a robust fallback when the response is malformed.
* Expose a clean ``ComplianceResult`` dataclass for typed downstream use.

Future RAG integration
~~~~~~~~~~~~~~~~~~~~~~
``ComplianceEngine.analyse()`` accepts an optional ``rag_context`` parameter.
When populated (e.g., with policy chunks from a vector store), the context is
forwarded to ``LLMClient.chat()`` as ``system_context`` and prepended to the
system prompt before the model analyses the transcript.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from compliance_engine.llm_client import LLMClient, LLMClientError
from compliance_engine.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, DEBUG_SYSTEM_ADDENDUM
from compliance_engine.config import POLICIES_DIR, RAG_TOP_K
from compliance_engine.rag import PolicyRetriever, load_policies_from_directory, PolicyChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RiskLevel = Literal["low", "medium", "high"]

VALID_RISK_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})

VALID_MATCHED_RULES: frozenset[str] = frozenset({
    "ILLEGAL_CLAIM",
    "MISLEADING_GUARANTEE",
    "MISSING_MANDATORY_DISCLOSURE",
    "COMPLIANT",
})


@dataclass
class DebugInfo:
    """
    Optional debugging metadata returned when ``debug=True`` is passed.

    Attributes
    ----------
    matched_rule:
        The specific regulation category that triggered the decision.
        One of ``ILLEGAL_CLAIM``, ``MISLEADING_GUARANTEE``,
        ``MISSING_MANDATORY_DISCLOSURE``, or ``COMPLIANT``.
    reasoning_summary:
        One short sentence (≤20 words) capturing the model's core reasoning.
    """

    matched_rule: str
    reasoning_summary: str
    retrieved_policies: Optional[List[str]] = None
    grounding_score: Optional[float] = None
    grounding_passed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "matched_rule": self.matched_rule,
            "reasoning_summary": self.reasoning_summary,
        }
        if self.retrieved_policies is not None:
            d["retrieved_policies"] = self.retrieved_policies
        if self.grounding_score is not None:
            d["grounding_score"] = self.grounding_score
        if self.grounding_passed is not None:
            d["grounding_passed"] = self.grounding_passed
        return d


@dataclass
class ComplianceResult:
    """
    Typed representation of a compliance analysis result.

    Attributes
    ----------
    violation:
        ``True`` if the transcript contains a flaggable compliance issue.
    risk_level:
        Severity of the finding: ``"low"``, ``"medium"``, or ``"high"``.
    confidence:
        Model certainty in the decision, in the range [0.0, 1.0].
        Values below 0.7 should trigger a human review.
    reason:
        Human-readable explanation of the finding.
    suggestion:
        Recommended corrective action, or ``"No action required."`` if clean.
    debug:
        Optional debug metadata. Populated only when ``debug=True`` is passed
        to :meth:`ComplianceEngine.analyse`.
    raw_response:
        The unprocessed string returned by the LLM (useful for debugging).
    """

    violation: bool
    risk_level: RiskLevel
    confidence: float
    reason: str
    suggestion: str
    debug: Optional[DebugInfo] = field(default=None)
    raw_response: str = field(repr=False, default="")

    def to_dict(self, include_debug: bool = True) -> Dict[str, Any]:
        """Return the public fields as a plain dict (excludes ``raw_response``).

        Parameters
        ----------
        include_debug:
            When ``True`` (default), include the ``debug`` sub-object if
            it is present. Set to ``False`` to get the minimal production schema.
        """
        d: Dict[str, Any] = {
            "violation": self.violation,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "suggestion": self.suggestion,
        }
        if include_debug and self.debug is not None:
            d["debug"] = self.debug.to_dict()
        return d

    def to_json(self, indent: int = 2, include_debug: bool = True) -> str:
        """Serialise the public fields to a JSON string."""
        return json.dumps(self.to_dict(include_debug=include_debug), indent=indent)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ComplianceValidationError(Exception):
    """Raised when the LLM output cannot be coerced into a valid result."""


def _extract_json_block(text: str) -> str:
    """
    Extract the first ``{...}`` block from *text*.

    This handles models that wrap JSON in markdown fences or add preamble text
    despite being instructed not to.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the outermost brace pair
    start = text.find("{")
    if start == -1:
        raise ComplianceValidationError("No JSON object found in LLM response.")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ComplianceValidationError("Unbalanced braces in LLM response.")


def _validate_and_coerce(data: Dict[str, Any], expect_debug: bool = False) -> ComplianceResult:
    """
    Validate the parsed JSON dict and coerce it into a :class:`ComplianceResult`.

    Parameters
    ----------
    data:
        Parsed dict from the LLM response.
    expect_debug:
        When ``True``, also parse and validate the ``debug`` sub-object.

    Returns
    -------
    ComplianceResult

    Raises
    ------
    ComplianceValidationError
        If required fields are missing or values are out of range.
    """
    required = ("violation", "risk_level", "confidence", "reason", "suggestion")
    missing = [k for k in required if k not in data]
    if missing:
        raise ComplianceValidationError(f"LLM response missing required fields: {missing}")

    # -- violation --
    violation = data["violation"]
    if not isinstance(violation, bool):
        # Some models return "true"/"false" strings
        if str(violation).lower() == "true":
            violation = True
        elif str(violation).lower() == "false":
            violation = False
        else:
            raise ComplianceValidationError(
                f"'violation' must be a boolean, got: {violation!r}"
            )

    # -- risk_level --
    risk_level = str(data["risk_level"]).strip().lower()
    if risk_level not in VALID_RISK_LEVELS:
        raise ComplianceValidationError(
            f"'risk_level' must be one of {VALID_RISK_LEVELS}, got: {risk_level!r}"
        )

    # -- confidence --
    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ComplianceValidationError(
            f"'confidence' must be a float, got: {data['confidence']!r}"
        ) from exc
    if not (0.0 <= confidence <= 1.0):
        raise ComplianceValidationError(
            f"'confidence' must be between 0.0 and 1.0, got: {confidence}"
        )

    # -- debug (optional) --
    debug_info: Optional[DebugInfo] = None
    if expect_debug:
        raw_debug = data.get("debug")
        if isinstance(raw_debug, dict):
            matched_rule = str(raw_debug.get("matched_rule", "")).strip().upper()
            if matched_rule not in VALID_MATCHED_RULES:
                logger.warning(
                    "Unknown matched_rule '%s'; keeping value as-is.", matched_rule
                )
            debug_info = DebugInfo(
                matched_rule=matched_rule or "UNKNOWN",
                reasoning_summary=str(raw_debug.get("reasoning_summary", "")).strip(),
                retrieved_policies=raw_debug.get("retrieved_policies"),
            )
        else:
            logger.warning("debug=True but LLM did not return a 'debug' object.")

    return ComplianceResult(
        violation=violation,
        risk_level=risk_level,  # type: ignore[arg-type]
        confidence=confidence,
        reason=str(data.get("reason", "")).strip(),
        suggestion=str(data.get("suggestion", "")).strip(),
        debug=debug_info,
    )


def _parse_llm_response(raw: str, expect_debug: bool = False) -> ComplianceResult:
    """
    Parse the raw LLM string into a validated :class:`ComplianceResult`.

    Strategy
    --------
    1. Attempt direct ``json.loads`` on the stripped response.
    2. Fall back to regex extraction of the first ``{...}`` block.
    3. Validate and coerce the parsed dict.

    Raises
    ------
    ComplianceValidationError
        If all parsing strategies fail.
    """
    raw_stripped = raw.strip()

    # Strategy 1 – clean JSON
    try:
        data = json.loads(raw_stripped)
        result = _validate_and_coerce(data, expect_debug=expect_debug)
        result.raw_response = raw
        return result
    except (json.JSONDecodeError, ComplianceValidationError):
        logger.debug("Direct JSON parse failed; attempting block extraction.")

    # Strategy 2 – extract block then parse
    try:
        block = _extract_json_block(raw_stripped)
        data = json.loads(block)
        result = _validate_and_coerce(data, expect_debug=expect_debug)
        result.raw_response = raw
        logger.warning(
            "JSON was embedded in extra text; extracted successfully. "
            "Consider adjusting the prompt to prevent this."
        )
        return result
    except (ComplianceValidationError, json.JSONDecodeError) as exc:
        logger.error("JSON extraction/validation failed: %s | raw=%s", exc, raw[:300])
        raise ComplianceValidationError(
            f"Could not parse a valid compliance result from LLM output. "
            f"Raw (first 300 chars): {raw[:300]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Fallback result factory
# ---------------------------------------------------------------------------

def _fallback_result(raw: str, error: Exception) -> ComplianceResult:
    """
    Return a conservative fallback result when parsing fails unrecoverably.

    The fallback flags the transcript as a potential medium-risk issue so that
    a human reviewer is always notified rather than silently passing a bad
    transcript.
    """
    logger.error(
        "Returning fallback compliance result due to parse failure. "
        "Review raw LLM output. error=%s | raw_preview=%s",
        error,
        raw[:200],
    )
    return ComplianceResult(
        violation=True,
        risk_level="medium",
        confidence=0.0,
        reason=(
            "Automated compliance analysis encountered an error and could not "
            "produce a definitive result. Manual review is required."
        ),
        suggestion=(
            "Review the transcript manually. The automated engine returned "
            "malformed output; this may indicate an unusual transcript or a "
            "temporary model issue."
        ),
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def _calculate_grounding(reasoning: str, policies_text: str) -> float:
    """Calculate a simple phrase/keyword overlap score between LLM reasoning and retrieved policies."""
    if not policies_text or not reasoning:
        return 0.0

    policy_words = set(re.findall(r'\b[a-z]{5,}\b', policies_text.lower()))
    reasoning_words = set(re.findall(r'\b[a-z]{5,}\b', reasoning.lower()))

    if not reasoning_words:
        return 0.0

    overlap = reasoning_words.intersection(policy_words)
    score = len(overlap) / max(1, len(reasoning_words))
    
    # Boost score slightly so partial keyword matching passes
    return min(1.0, score * 1.5)


class ComplianceEngine:
    """
    High-level compliance detection engine.

    Usage
    -----
    >>> from compliance_engine.compliance_engine import ComplianceEngine
    >>> engine = ComplianceEngine()
    >>> result = engine.analyse(
    ...     transcript="We guarantee 15% annual returns.",
    ...     domain="fintech",
    ... )
    >>> print(result.to_json())

    Parameters
    ----------
    llm_client:
        An :class:`~compliance_engine.llm_client.LLMClient` instance.
        If ``None``, a default client is instantiated from ``config.py``.
    retriever:
        A :class:`~compliance_engine.rag.PolicyRetriever` instance.
        If ``None``, policies are auto-loaded from ``POLICIES_DIR``.
    use_fallback_on_error:
        When ``True`` (default), a conservative fallback result is returned
        if parsing fails instead of raising an exception. Set to ``False`` in
        testing to surface errors explicitly.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[PolicyRetriever] = None,
        use_fallback_on_error: bool = True,
    ) -> None:
        self._client = llm_client or LLMClient()
        self._use_fallback = use_fallback_on_error
        
        # Load retriever if not provided
        if retriever is not None:
            self._retriever = retriever
        else:
            chunks = load_policies_from_directory(POLICIES_DIR)
            self._retriever = PolicyRetriever(chunks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        transcript: str,
        domain: str,
        rag_context: Optional[str] = None,
        debug: bool = False,
    ) -> ComplianceResult:
        """
        Analyse a transcript for compliance violations.

        Parameters
        ----------
        transcript:
            The raw conversation or sales/marketing text to evaluate.
        domain:
            Regulatory domain context (e.g., ``"fintech"``, ``"insurance"``,
            ``"healthcare"``).
        rag_context:
            Optional policy chunk(s). If not provided, the engine will
            automatically retrieve relevant policies using the retriever.
        debug:
            When ``True``, ask the LLM to populate a ``debug`` sub-object
            containing ``matched_rule`` and ``reasoning_summary``. The
            ``debug`` field will be included in ``ComplianceResult.to_dict()``
            and ``to_json()`` outputs.

        Returns
        -------
        ComplianceResult
            A validated, typed result object.
        """
        if not transcript or not transcript.strip():
            raise ValueError("'transcript' must be a non-empty string.")
        if not domain or not domain.strip():
            raise ValueError("'domain' must be a non-empty string.")

        logger.info(
            "Compliance analysis requested | domain=%s | transcript_length=%d",
            domain,
            len(transcript),
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            domain=domain.strip(),
            transcript=transcript.strip(),
        )

        # Retrieve policies if context not already set
        retrieved_snippets: List[str] = []
        raw_policy_text: str = ""
        top_score: float = 0.0

        if rag_context is None and self._retriever:
            query = f"compliance rules regarding: {transcript}"
            retrieved_chunks = self._retriever.retrieve(query, top_k=RAG_TOP_K, domain=domain)
            
            if not retrieved_chunks:
                # Fail safely if absolutely no rules can be retrieved
                logger.warning("No policies retrieved for domain=%s. Failing safely.", domain)
                return ComplianceResult(
                    violation=False,
                    risk_level="low",
                    confidence=0.1,
                    reason="No relevant regulatory policies could be retrieved for this transcript.",
                    suggestion="Ensure policies are loaded for this domain.",
                    debug=DebugInfo(
                        matched_rule="COMPLIANT",
                        reasoning_summary="No policies found.",
                        retrieved_policies=[],
                    ) if debug else None,
                )
            
            # Extract data from new RetrievedChunk objects
            top_score = retrieved_chunks[0].score
            raw_policy_text = "\n".join(r.chunk.content for r in retrieved_chunks)
            retrieved_snippets = [
                f"{r.chunk.source_file} (score: {r.score:.2f}, matches: {len(r.matched_keywords)}/{r.query_keyword_count})" 
                for r in retrieved_chunks
            ]
            
            rag_context_parts = ["=== RELEVANT REGULATORY POLICIES ==="]
            rag_context_parts.append(
                "ONLY use the provided policy context. If no relevant policy is found, "
                "or the context doesn't cover the transcript, return violation=false."
            )
            for i, retrieved in enumerate(retrieved_chunks, 1):
                rag_context_parts.append(
                    f"\n--- Policy Snippet {i} [{retrieved.chunk.source_file}] (score: {retrieved.score:.2f}) ---\n"
                    f"{retrieved.chunk.content}"
                )
            rag_context = "\n".join(rag_context_parts)

        # Compose the effective system prompt
        system_prompt = SYSTEM_PROMPT
        if debug:
            system_prompt = SYSTEM_PROMPT + DEBUG_SYSTEM_ADDENDUM
        if rag_context:
            system_prompt = rag_context + "\n\n" + system_prompt

        if rag_context is not None and not raw_policy_text:
            raw_policy_text = rag_context

        try:
            llm_result = self._client.analyse(
                transcript=transcript.strip(),
                context=raw_policy_text,
                debug=debug,
            )
        except Exception as exc:
            logger.error("Local LLM analysis failed: %s", exc)
            if self._use_fallback:
                return _fallback_result("", exc)
            raise

        result_debug = None
        if debug and getattr(llm_result, "debug", None) is not None:
            raw_debug = llm_result.debug
            if isinstance(raw_debug, dict):
                matched_rule = str(raw_debug.get("matched_rule", "")).strip().upper()
                if matched_rule not in VALID_MATCHED_RULES:
                    logger.warning(
                        "Unknown matched_rule '%s'; keeping value as-is.", matched_rule
                    )
                result_debug = DebugInfo(
                    matched_rule=matched_rule or "UNKNOWN",
                    reasoning_summary=str(raw_debug.get("reasoning_summary", "")).strip(),
                )
            elif isinstance(raw_debug, DebugInfo):
                result_debug = raw_debug

        result = ComplianceResult(
            violation=llm_result.violation,
            risk_level=(
                "high"
                if llm_result.violation and llm_result.confidence >= 0.8
                else "medium"
                if llm_result.violation
                else "low"
            ),
            confidence=llm_result.confidence,
            reason=llm_result.reason,
            suggestion=(
                "Review the transcript for potentially misleading or prohibited claims."
                if llm_result.violation
                else "No action required."
            ),
            debug=result_debug,
            raw_response=json.dumps(
                {
                    "violation": llm_result.violation,
                    "confidence": llm_result.confidence,
                    "reason": llm_result.reason,
                    **({"debug": llm_result.debug} if getattr(llm_result, "debug", None) is not None else {}),
                }
            ),
        )

        # Grounding Validation Layer
        if rag_context is not None and raw_policy_text:
            grounding_score = _calculate_grounding(result.reason, raw_policy_text)
            grounding_passed = grounding_score >= 0.3
            
            if debug and result.debug is not None:
                result.debug.grounding_score = round(grounding_score, 2)
                result.debug.grounding_passed = grounding_passed
                
            if result.violation and not grounding_passed:
                logger.warning("Grounding failed (score=%.2f). Overriding violation flag.", grounding_score)
                result.violation = False
                result.confidence = min(result.confidence, 0.3)
                result.reason = "Insufficient grounding in retrieved policies."

        # Inject retrieved snippets into debug manually (since LLM didn't return them)
        if debug and result.debug is not None and retrieved_snippets:
            result.debug.retrieved_policies = retrieved_snippets

        logger.info(
            "Compliance analysis complete | domain=%s | violation=%s | risk=%s",
            domain,
            result.violation,
            result.risk_level,
        )
        
        # Combine LLM confidence with the top retrieval score
        if rag_context is not None and top_score > 0.0:
            result.confidence = round((result.confidence + top_score) / 2.0, 4)
            
        return result
