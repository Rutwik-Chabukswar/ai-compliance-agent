"""
tests/test_compliance_engine.py
-------------------------------
Unit tests for the compliance detection engine.

Run with:
    pytest -v tests/
"""

import json
import pytest
from unittest.mock import MagicMock

from compliance_engine.compliance_engine import (
    ComplianceEngine,
    ComplianceResult,
    ComplianceValidationError,
    DebugInfo,
    _extract_json_block,
    _parse_llm_response,
    _validate_and_coerce,
)
from compliance_engine.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm_client() -> MagicMock:
    """Return a MagicMock that satisfies the LLMClient interface."""
    return MagicMock(spec=LLMClient)


@pytest.fixture()
def mock_retriever() -> MagicMock:
    """Return a mock retriever that returns a dummy chunk and score."""
    from compliance_engine.rag import PolicyChunk, PolicyRetriever
    retriever = MagicMock(spec=PolicyRetriever)
    chunk = PolicyChunk(source_file="dummy.txt", content="Guaranteed returns promise is illegal violation.", domain="fintech")
    retriever.retrieve.return_value = [(chunk, 0.3)]
    return retriever


@pytest.fixture()
def engine(mock_llm_client: MagicMock, mock_retriever: MagicMock) -> ComplianceEngine:
    return ComplianceEngine(llm_client=mock_llm_client, retriever=mock_retriever, use_fallback_on_error=False)


def _make_raw(
    violation: bool,
    risk: str,
    reason: str,
    suggestion: str,
    confidence: float = 0.95,
    debug: dict | None = None,
) -> str:
    payload: dict = {
        "violation": violation,
        "risk_level": risk,
        "confidence": confidence,
        "reason": reason,
        "suggestion": suggestion,
    }
    if debug is not None:
        payload["debug"] = debug
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# _extract_json_block
# ---------------------------------------------------------------------------

class TestExtractJsonBlock:
    def test_clean_json(self):
        raw = '{"violation": false, "risk_level": "low", "confidence": 0.9, "reason": "ok", "suggestion": "none"}'
        assert _extract_json_block(raw) == raw

    def test_json_with_preamble(self):
        raw = 'Here is the output: {"violation": true, "risk_level": "high", "confidence": 0.95, "reason": "bad", "suggestion": "fix it"}'
        block = _extract_json_block(raw)
        assert block.startswith("{")
        assert json.loads(block)["violation"] is True

    def test_markdown_fence_stripped(self):
        raw = '```json\n{"violation": false, "risk_level": "low", "confidence": 0.9, "reason": "-", "suggestion": "-"}\n```'
        block = _extract_json_block(raw)
        assert json.loads(block)["violation"] is False

    def test_no_json_raises(self):
        with pytest.raises(ComplianceValidationError, match="No JSON object found"):
            _extract_json_block("This is just plain text.")

    def test_unbalanced_braces_raises(self):
        with pytest.raises(ComplianceValidationError, match="Unbalanced braces"):
            _extract_json_block('{"violation": true, "risk_level": "high"')


# ---------------------------------------------------------------------------
# _validate_and_coerce
# ---------------------------------------------------------------------------

class TestValidateAndCoerce:
    def test_valid_dict(self):
        d = {"violation": True, "risk_level": "high", "confidence": 0.95, "reason": "bad", "suggestion": "fix"}
        result = _validate_and_coerce(d)
        assert result.violation is True
        assert result.risk_level == "high"
        assert result.confidence == 0.95

    def test_string_booleans_coerced(self):
        d = {"violation": "true", "risk_level": "low", "confidence": 0.8, "reason": "ok", "suggestion": "-"}
        result = _validate_and_coerce(d)
        assert result.violation is True

    def test_invalid_risk_level_raises(self):
        d = {"violation": False, "risk_level": "critical", "confidence": 0.5, "reason": "x", "suggestion": "x"}
        with pytest.raises(ComplianceValidationError, match="risk_level"):
            _validate_and_coerce(d)

    def test_missing_field_raises(self):
        d = {"violation": False, "risk_level": "low"}
        with pytest.raises(ComplianceValidationError, match="missing required fields"):
            _validate_and_coerce(d)

    def test_invalid_boolean_raises(self):
        d = {"violation": "yes", "risk_level": "low", "confidence": 0.8, "reason": "x", "suggestion": "x"}
        with pytest.raises(ComplianceValidationError, match="violation"):
            _validate_and_coerce(d)

    def test_missing_confidence_raises(self):
        d = {"violation": False, "risk_level": "low", "reason": "x", "suggestion": "x"}
        with pytest.raises(ComplianceValidationError, match="missing required fields"):
            _validate_and_coerce(d)

    def test_confidence_out_of_range_raises(self):
        d = {"violation": False, "risk_level": "low", "confidence": 1.5, "reason": "x", "suggestion": "x"}
        with pytest.raises(ComplianceValidationError, match="confidence"):
            _validate_and_coerce(d)

    def test_confidence_boundary_values(self):
        for val in (0.0, 1.0):
            d = {"violation": False, "risk_level": "low", "confidence": val, "reason": "x", "suggestion": "x"}
            result = _validate_and_coerce(d)
            assert result.confidence == val

    def test_debug_parsed_when_expect_debug_true(self):
        d = {
            "violation": True,
            "risk_level": "high",
            "confidence": 0.95,
            "reason": "bad",
            "suggestion": "fix",
            "debug": {
                "matched_rule": "MISLEADING_GUARANTEE",
                "reasoning_summary": "Guaranteed return claim present.",
            },
        }
        result = _validate_and_coerce(d, expect_debug=True)
        assert result.debug is not None
        assert result.debug.matched_rule == "MISLEADING_GUARANTEE"
        assert "Guaranteed" in result.debug.reasoning_summary

    def test_debug_none_when_expect_debug_false(self):
        d = {
            "violation": False,
            "risk_level": "low",
            "confidence": 0.9,
            "reason": "ok",
            "suggestion": "none",
            "debug": {"matched_rule": "COMPLIANT", "reasoning_summary": "Fine."},
        }
        result = _validate_and_coerce(d, expect_debug=False)
        assert result.debug is None

    def test_debug_missing_block_logs_warning(self, caplog):
        import logging
        d = {"violation": False, "risk_level": "low", "confidence": 0.9, "reason": "ok", "suggestion": "none"}
        with caplog.at_level(logging.WARNING, logger="compliance_engine.compliance_engine"):
            result = _validate_and_coerce(d, expect_debug=True)
        assert result.debug is None
        assert any("debug" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_clean_json_parses(self):
        raw = _make_raw(False, "low", "All good.", "No action required.")
        result = _parse_llm_response(raw)
        assert result.violation is False
        assert result.risk_level == "low"
        assert result.confidence == 0.95

    def test_json_with_markdown_fence(self):
        raw = "```json\n" + _make_raw(True, "high", "Bad claim.", "Remove it.") + "\n```"
        result = _parse_llm_response(raw)
        assert result.violation is True

    def test_json_with_trailing_text(self):
        raw = _make_raw(True, "medium", "Maybe bad.", "Review it.") + " Additional notes here."
        result = _parse_llm_response(raw)
        assert result.risk_level == "medium"

    def test_completely_invalid_response_raises(self):
        with pytest.raises(ComplianceValidationError):
            _parse_llm_response("I cannot analyse this. Please try again.")

    def test_debug_parsed_when_expect_debug(self):
        raw = _make_raw(
            True, "high", "Violation.", "Fix it.", 0.92,
            debug={"matched_rule": "ILLEGAL_CLAIM", "reasoning_summary": "Fabricated claim found."},
        )
        result = _parse_llm_response(raw, expect_debug=True)
        assert result.debug is not None
        assert result.debug.matched_rule == "ILLEGAL_CLAIM"


# ---------------------------------------------------------------------------
# ComplianceEngine.analyse
# ---------------------------------------------------------------------------

class TestComplianceEngineAnalyse:
    def test_violation_detected(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(
            True, "high", "Guaranteed returns promise is illegal.", "Remove it."
        )
        result = engine.analyse("We guarantee 15% returns.", "fintech")
        assert result.violation is True
        assert result.risk_level == "high"
        assert result.confidence == 0.625

    def test_compliant_transcript(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(
            False, "low", "Balanced with disclaimers.", "No action required.", 0.98
        )
        result = engine.analyse(
            "Historically our fund has returned 8%. Past performance is not a guarantee.",
            "fintech",
        )
        assert result.violation is False
        assert result.suggestion == "No action required."

    def test_empty_transcript_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.analyse("", "fintech")

    def test_empty_domain_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.analyse("Some transcript.", "")

    def test_debug_mode_returns_debug_info(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(
            True, "high", "Violation.", "Fix it.", 0.95,
            debug={"matched_rule": "MISLEADING_GUARANTEE", "reasoning_summary": "Return guarantee present."},
        )
        result = engine.analyse("...", "fintech", debug=True)
        assert result.debug is not None
        assert result.debug.matched_rule == "MISLEADING_GUARANTEE"

    def test_debug_mode_false_returns_no_debug(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(False, "low", "OK", "No action required.")
        result = engine.analyse("...", "fintech", debug=False)
        assert result.debug is None

    def test_debug_prompt_appended_when_debug_true(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(False, "low", "OK", "No action required.")
        engine.analyse("Transcript.", "fintech", debug=True)
        call_kwargs = mock_llm_client.chat.call_args[1]
        assert "DEBUG MODE" in call_kwargs["system_prompt"]

    def test_rag_context_prepended_to_system_prompt(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(False, "low", "OK", "No action required.")
        engine.analyse("Transcript.", "insurance", rag_context="Policy chunk here.")
        call_kwargs = mock_llm_client.chat.call_args[1]
        assert call_kwargs["system_prompt"].startswith("Policy chunk here.")

    def test_fallback_on_parse_error(self, mock_llm_client, mock_retriever):
        """Engine in fallback mode returns medium-risk result instead of raising."""
        mock_llm_client.chat.return_value = "This is not JSON at all."
        safe_engine = ComplianceEngine(llm_client=mock_llm_client, retriever=mock_retriever, use_fallback_on_error=True)
        result = safe_engine.analyse("Some transcript.", "fintech")
        assert result.violation is True
        assert result.risk_level == "medium"
        assert result.confidence == 0.0
        assert "Manual review" in result.reason

    def test_to_dict_includes_confidence(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(False, "low", "OK", "No action required.", 0.88)
        result = engine.analyse("Transcript.", "fintech")
        d = result.to_dict()
        assert "confidence" in d
        assert d["confidence"] == 0.59

    def test_to_dict_excludes_raw_response(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(False, "low", "OK", "No action required.")
        result = engine.analyse("Transcript.", "fintech")
        d = result.to_dict()
        assert "raw_response" not in d
        assert set(d.keys()) == {"violation", "risk_level", "confidence", "reason", "suggestion"}

    def test_to_dict_includes_debug_when_present(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(
            False, "low", "OK", "No action required.", 0.9,
            debug={"matched_rule": "COMPLIANT", "reasoning_summary": "All fine."},
        )
        result = engine.analyse("Transcript.", "fintech", debug=True)
        d = result.to_dict()
        assert "debug" in d
        assert d["debug"]["matched_rule"] == "COMPLIANT"

    def test_to_dict_excludes_debug_when_include_debug_false(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(
            False, "low", "OK", "No action required.", 0.9,
            debug={"matched_rule": "COMPLIANT", "reasoning_summary": "Fine."},
        )
        result = engine.analyse("Transcript.", "fintech", debug=True)
        d = result.to_dict(include_debug=False)
        assert "debug" not in d

    def test_to_json_is_valid_json(self, engine, mock_llm_client):
        mock_llm_client.chat.return_value = _make_raw(True, "high", "Violation.", "Fix it.")
        result = engine.analyse("Bad claim.", "fintech")
        parsed = json.loads(result.to_json())
        assert parsed["violation"] is True
        assert "confidence" in parsed


# ---------------------------------------------------------------------------
# DebugInfo
# ---------------------------------------------------------------------------

class TestDebugInfo:
    def test_to_dict(self):
        dbg = DebugInfo(matched_rule="ILLEGAL_CLAIM", reasoning_summary="False cure claim.")
        d = dbg.to_dict()
        assert d == {"matched_rule": "ILLEGAL_CLAIM", "reasoning_summary": "False cure claim."}
