"""
agent.py
--------
Lightweight agent orchestration layer for compliance detection.

The ComplianceAgent dynamically decides which tools to use based on
transcript content and executes them in sequence to produce a final
compliance decision.

Strategies:
  - "rule_only": Fast rule-based detection for obvious violations
  - "direct_llm": Direct LLM analysis without external context
  - "rag_augmented": LLM analysis augmented with retrieved policies

Design principles:
  - Small, focused functions
  - Comprehensive logging for visibility
  - Type hints and docstrings
  - No external agent frameworks (keep it simple)
  - Seamless integration with existing ComplianceEngine
"""

import logging
import re
from typing import List, Optional, Literal

from compliance_engine.compliance_engine import ComplianceEngine, ComplianceResult
from compliance_engine.llm_client import LLMClient, LLMResponse
from compliance_engine.rag import PolicyRetriever, RetrievedChunk
from compliance_engine.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# =========================================================================
# Strategy Detection Keywords
# =========================================================================

# High-confidence indicators of rule-based violations
RULE_ONLY_KEYWORDS = {
    "guarantee",
    "guaranteed",
    "guarantees",
    "100%",
    "risk-free",
    "riskfree",
    "zero risk",
    "no risk",
}

# Indicators that RAG augmentation would be beneficial
RAG_AUGMENTED_KEYWORDS = {
    "investment",
    "policy",
    "policies",
    "returns",
    "insurance",
    "premium",
    "coverage",
    "compliance",
    "regulation",
    "sec",
    "finra",
}

# =========================================================================
# Strategy Decision
# =========================================================================


def _decide_strategy(transcript: str) -> Literal["rule_only", "rag_augmented", "direct_llm"]:
    """
    Decide which execution strategy to use based on transcript content.

    Logic:
    1. If transcript contains RULE_ONLY_KEYWORDS → "rule_only"
    2. Else if transcript contains RAG_AUGMENTED_KEYWORDS → "rag_augmented"
    3. Else → "direct_llm"

    Parameters
    ----------
    transcript : str
        The transcript to analyze.

    Returns
    -------
    Literal["rule_only", "rag_augmented", "direct_llm"]
        The selected strategy.
    """
    # Normalize for matching
    text_lower = transcript.lower()
    
    # First check for patterns with special characters (100%, risk-free, etc.) before normalization
    if "100%" in text_lower:
        logger.debug("Strategy decision: RULE_ONLY (keyword='100%')")
        return "rule_only"
    if "risk-free" in text_lower or "riskfree" in text_lower:
        logger.debug("Strategy decision: RULE_ONLY (keyword='risk-free')")
        return "rule_only"
    if "zero risk" in text_lower:
        logger.debug("Strategy decision: RULE_ONLY (keyword='zero risk')")
        return "rule_only"
    if "no risk" in text_lower:
        logger.debug("Strategy decision: RULE_ONLY (keyword='no risk')")
        return "rule_only"

    # Normalize non-alphanumeric characters to spaces for word matching
    text_normalized = re.sub(r"[^\w\s]", " ", text_lower)

    # Check for high-confidence rule violations first
    for keyword in RULE_ONLY_KEYWORDS:
        if keyword in text_normalized:
            logger.debug(
                "Strategy decision: RULE_ONLY (keyword='%s')", keyword
            )
            return "rule_only"

    # Check for domain-specific keywords
    for keyword in RAG_AUGMENTED_KEYWORDS:
        if keyword in text_normalized:
            logger.debug(
                "Strategy decision: RAG_AUGMENTED (keyword='%s')", keyword
            )
            return "rag_augmented"

    # Default to direct LLM
    logger.debug("Strategy decision: DIRECT_LLM (no special keywords detected)")
    return "direct_llm"


# =========================================================================
# Strategy Implementations
# =========================================================================


def _run_rules(
    llm_client: LLMClient,
    transcript: str,
) -> ComplianceResult:
    """
    Execute rule-only strategy using LLM fallback engine.

    This is the fastest path - directly invokes the rule-based fallback
    without LLM overhead or RAG retrieval.

    Parameters
    ----------
    llm_client : LLMClient
        LLM client instance (used for fallback engine).
    transcript : str
        Transcript to analyze.

    Returns
    -------
    ComplianceResult
        Compliance decision from rule engine.
    """
    logger.debug("Executing RULE_ONLY strategy (transcript_len=%d)", len(transcript))

    # Get fallback rule engine result
    rule_result_dict = llm_client._fallback_rule_engine(transcript)

    # Convert to LLMResponse for consistency
    llm_response = LLMResponse(
        violation=rule_result_dict["violation"],
        confidence=rule_result_dict["confidence"],
        reason=rule_result_dict["reason"],
        debug={
            "matched_rule": "RULE_BASED_DETECTION",
            "reasoning_summary": rule_result_dict["reason"][:100],
        },
    )

    logger.info(
        "RULE_ONLY strategy completed: violation=%s, confidence=%.2f",
        llm_response.violation,
        llm_response.confidence,
    )

    # Convert to ComplianceResult
    result = ComplianceResult(
        violation=llm_response.violation,
        risk_level="high" if llm_response.violation else "low",
        confidence=llm_response.confidence,
        reason=llm_response.reason,
        suggestion=rule_result_dict.get("suggestion", "No action required."),
    )

    return result


def _run_llm(
    llm_client: LLMClient,
    transcript: str,
    domain: str,
) -> ComplianceResult:
    """
    Execute direct LLM strategy without external context.

    Calls the LLM directly with standard system prompt and transcript.
    No RAG retrieval, no augmentation. Clean and straightforward.

    Parameters
    ----------
    llm_client : LLMClient
        LLM client instance.
    transcript : str
        Transcript to analyze.
    domain : str
        Domain context (e.g., "fintech", "insurance").

    Returns
    -------
    ComplianceResult
        Compliance decision from LLM.
    """
    logger.debug(
        "Executing DIRECT_LLM strategy (transcript_len=%d, domain=%s)",
        len(transcript),
        domain,
    )

    system_prompt = f"{SYSTEM_PROMPT}\n\nDomain: {domain}"

    # Call LLM with analyse() wrapper
    llm_response = llm_client.analyse(transcript=transcript, context=None)

    logger.info(
        "DIRECT_LLM strategy completed: violation=%s, confidence=%.2f",
        llm_response.violation,
        llm_response.confidence,
    )

    # Convert to ComplianceResult
    result = ComplianceResult(
        violation=llm_response.violation,
        risk_level="high" if llm_response.violation else "low",
        confidence=llm_response.confidence,
        reason=llm_response.reason,
        suggestion="No action required." if not llm_response.violation else "Review and revise content.",
    )

    return result


def _run_rag_pipeline(
    llm_client: LLMClient,
    retriever: PolicyRetriever,
    transcript: str,
    domain: str,
    top_k: int = 3,
) -> ComplianceResult:
    """
    Execute RAG-augmented strategy using retrieved policies.

    1. Retrieve top-k policies relevant to transcript
    2. Inject policy context into system prompt
    3. Call LLM with augmented prompt
    4. Return compliance decision

    Parameters
    ----------
    llm_client : LLMClient
        LLM client instance.
    retriever : PolicyRetriever
        Policy retriever for RAG lookups.
    transcript : str
        Transcript to analyze.
    domain : str
        Domain context for filtering (e.g., "fintech").
    top_k : int
        Number of policies to retrieve (default 3).

    Returns
    -------
    ComplianceResult
        Compliance decision from LLM with RAG context.
    """
    logger.debug(
        "Executing RAG_AUGMENTED strategy (transcript_len=%d, domain=%s, top_k=%d)",
        len(transcript),
        domain,
        top_k,
    )

    # Retrieve relevant policies
    retrieved_chunks: List[RetrievedChunk] = retriever.retrieve(
        query=transcript,
        top_k=top_k,
        domain=domain,
    )

    logger.debug("Retrieved %d policy chunks from RAG", len(retrieved_chunks))

    # Format retrieved policies for prompt injection
    policy_context = _format_rag_context(retrieved_chunks)

    # Augment system prompt with policy context
    augmented_system_prompt = f"{SYSTEM_PROMPT}\n\n{policy_context}\nDomain: {domain}"

    # Call LLM with augmented context
    llm_response = llm_client.analyse(
        transcript=transcript,
        context=policy_context,
    )

    # Calculate grounding score (confidence boost from RAG)
    grounding_score = _calculate_grounding_score(retrieved_chunks)

    logger.info(
        "RAG_AUGMENTED strategy completed: violation=%s, confidence=%.2f, grounding=%.2f, policies=%d",
        llm_response.violation,
        llm_response.confidence,
        grounding_score,
        len(retrieved_chunks),
    )

    # Blend LLM confidence with grounding score
    blended_confidence = (llm_response.confidence + grounding_score) / 2.0

    # Convert to ComplianceResult
    result = ComplianceResult(
        violation=llm_response.violation,
        risk_level="high" if llm_response.violation else "low",
        confidence=blended_confidence,
        reason=llm_response.reason,
        suggestion="Review relevant policies." if llm_response.violation else "No action required.",
        debug={
            "matched_rule": "RAG_AUGMENTED_LLM",
            "reasoning_summary": llm_response.reason[:100],
            "retrieved_policies": [
                f"{r.chunk.source_file} (score={r.score:.2f})"
                for r in retrieved_chunks
            ],
            "grounding_score": round(grounding_score, 4),
        } if retrieved_chunks else None,
    )

    return result


# =========================================================================
# Helper Functions
# =========================================================================


def _format_rag_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved policy chunks into a clear prompt context.

    Produces a readable policy section for injection into the system prompt.

    Parameters
    ----------
    retrieved_chunks : List[RetrievedChunk]
        Retrieved policy chunks from RAG.

    Returns
    -------
    str
        Formatted policy context string.
    """
    if not retrieved_chunks:
        return "No relevant policies retrieved."

    lines = ["=== RELEVANT REGULATORY POLICIES ==="]

    for i, retrieved in enumerate(retrieved_chunks, 1):
        lines.append(f"\n[Policy {i}] {retrieved.chunk.source_file}")
        lines.append(f"Score: {retrieved.score:.2f} | Matches: {len(retrieved.matched_keywords)}")
        lines.append(f"Content: {retrieved.chunk.content[:200]}...")

    lines.append("\nONLY use the provided policy context for your analysis.")

    return "\n".join(lines)


def _calculate_grounding_score(retrieved_chunks: List[RetrievedChunk]) -> float:
    """
    Calculate a grounding score based on retrieval relevance.

    Combines:
    - Average score of retrieved chunks
    - Penalty if no chunks retrieved
    - Normalized to [0.0, 1.0] range

    Parameters
    ----------
    retrieved_chunks : List[RetrievedChunk]
        Retrieved policy chunks.

    Returns
    -------
    float
        Grounding confidence score in [0.0, 1.0].
    """
    if not retrieved_chunks:
        return 0.0

    # Average score of retrieved chunks
    avg_score = sum(r.score for r in retrieved_chunks) / len(retrieved_chunks)

    # Boost score slightly for deep retrieval (more policies = more grounding)
    retrieval_boost = min(len(retrieved_chunks) * 0.05, 0.2)

    return min(avg_score + retrieval_boost, 1.0)


# =========================================================================
# Main Agent
# =========================================================================


class ComplianceAgent:
    """
    Lightweight orchestrator for compliance detection strategies.

    Dynamically selects and executes the most appropriate strategy for
    analyzing compliance violations in transcripts.

    Parameters
    ----------
    llm_client : LLMClient
        LLM client for analysis.
    retriever : Optional[PolicyRetriever]
        Policy retriever for RAG. If None, RAG strategies will be skipped.
    use_fallback_on_error : bool
        If True, fall back to rule-based detection on errors. Default True.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Optional[PolicyRetriever] = None,
        use_fallback_on_error: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self.use_fallback_on_error = use_fallback_on_error

        logger.info(
            "ComplianceAgent initialized (llm_available=%s, rag_available=%s)",
            llm_client.llm_available,
            retriever is not None,
        )

    def analyse(
        self,
        transcript: str,
        domain: str,
    ) -> ComplianceResult:
        """
        Analyze a transcript for compliance violations using dynamic strategy selection.

        Workflow:
        1. Decide which strategy to use based on transcript content
        2. Execute the selected strategy
        3. Return structured ComplianceResult

        Parameters
        ----------
        transcript : str
            The transcript to analyze (non-empty).
        domain : str
            Domain context (e.g., "fintech", "insurance").

        Returns
        -------
        ComplianceResult
            Compliance analysis result with violation status, risk level, and confidence.

        Raises
        ------
        ValueError
            If transcript or domain is empty/invalid.
        """
        # Validate inputs
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        if not domain or not domain.strip():
            raise ValueError("Domain cannot be empty")

        transcript = transcript.strip()
        domain = domain.strip()

        logger.info("ComplianceAgent.analyse() called (domain=%s, transcript_len=%d)", domain, len(transcript))

        try:
            # Decide strategy
            strategy = _decide_strategy(transcript)
            logger.info("Selected strategy: %s", strategy)

            # Execute strategy
            if strategy == "rule_only":
                result = _run_rules(self.llm_client, transcript)

            elif strategy == "direct_llm":
                result = _run_llm(self.llm_client, transcript, domain)

            elif strategy == "rag_augmented":
                if self.retriever is None:
                    logger.warning(
                        "Strategy is RAG_AUGMENTED but no retriever available. "
                        "Falling back to DIRECT_LLM."
                    )
                    result = _run_llm(self.llm_client, transcript, domain)
                else:
                    result = _run_rag_pipeline(
                        self.llm_client,
                        self.retriever,
                        transcript,
                        domain,
                    )
            else:
                # Should never reach here, but handle gracefully
                logger.error("Unknown strategy: %s. Falling back to direct LLM.", strategy)
                result = _run_llm(self.llm_client, transcript, domain)

            logger.info(
                "ComplianceAgent analysis complete: violation=%s, risk=%s, confidence=%.2f, strategy=%s",
                result.violation,
                result.risk_level,
                result.confidence,
                strategy,
            )

            return result

        except Exception as e:
            logger.error("ComplianceAgent analysis failed: %s", str(e), exc_info=True)

            if self.use_fallback_on_error:
                logger.warning("Falling back to rule-based detection")
                return _run_rules(self.llm_client, transcript)
            else:
                raise
