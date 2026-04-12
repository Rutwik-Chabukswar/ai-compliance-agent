"""
tests/test_agent.py
-------------------
Unit tests for the ComplianceAgent orchestration layer.

Tests cover:
  - Strategy selection logic
  - Individual strategy execution
  - RAG pipeline integration
  - Error handling and fallback behavior
"""

import pytest
from unittest.mock import MagicMock

from compliance_engine.agent import (
    ComplianceAgent,
    _decide_strategy,
    _run_rules,
    _run_llm,
    _run_rag_pipeline,
    _format_rag_context,
    _calculate_grounding_score,
)
from compliance_engine.compliance_engine import ComplianceResult
from compliance_engine.llm_client import LLMClient, LLMResponse
from compliance_engine.rag import PolicyChunk, PolicyRetriever, RetrievedChunk


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def mock_llm_client() -> MagicMock:
    """Mock LLMClient with both LLM and rule-based modes."""
    client = MagicMock(spec=LLMClient)
    client.llm_available = True
    client.use_fallback_on_error = True

    # Mock analyse() to return LLMResponse
    client.analyse.return_value = LLMResponse(
        violation=False,
        confidence=0.85,
        reason="No violations detected.",
    )

    # Mock _fallback_rule_engine() for rule-based strategy
    client._fallback_rule_engine.return_value = {
        "violation": False,
        "risk_level": "low",
        "confidence": 0.90,
        "reason": "Rule-based: No risk keywords detected.",
        "suggestion": "No action required.",
    }

    return client


@pytest.fixture()
def mock_retriever() -> MagicMock:
    """Mock PolicyRetriever with sample chunks."""
    retriever = MagicMock(spec=PolicyRetriever)

    # Create mock retrieved chunks
    chunk = PolicyChunk(
        source_file="fintech_policy.txt",
        content="Investment guarantees are prohibited under SEC regulations.",
        domain="fintech",
    )
    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.85,
        matched_keywords=["investment", "guarantee"],
        query_keyword_count=2,
    )

    retriever.retrieve.return_value = [retrieved]
    return retriever


@pytest.fixture()
def agent(mock_llm_client: MagicMock, mock_retriever: MagicMock) -> ComplianceAgent:
    """Create a ComplianceAgent with mocked dependencies."""
    return ComplianceAgent(
        llm_client=mock_llm_client,
        retriever=mock_retriever,
        use_fallback_on_error=True,
    )


# =========================================================================
# Strategy Selection Tests
# =========================================================================


class TestStrategyDecision:
    """Test the _decide_strategy decision logic."""

    def test_rule_only_guarantee_keyword(self):
        """Should select RULE_ONLY for guaranteed/guarantee keyword."""
        strategy = _decide_strategy("Our investment fund guarantees 10% returns.")
        assert strategy == "rule_only"

    def test_rule_only_100_percent_keyword(self):
        """Should select RULE_ONLY for 100% keyword."""
        strategy = _decide_strategy("You can earn 100% profit with zero effort.")
        assert strategy == "rule_only"

    def test_rule_only_risk_free_keyword(self):
        """Should select RULE_ONLY for risk-free keyword."""
        strategy = _decide_strategy("This product is completely risk-free.")
        assert strategy == "rule_only"

    def test_rag_augmented_investment_keyword(self):
        """Should select RAG_AUGMENTED for investment keyword."""
        strategy = _decide_strategy("Tell me about investment returns and policies.")
        assert strategy == "rag_augmented"

    def test_rag_augmented_insurance_keyword(self):
        """Should select RAG_AUGMENTED for insurance keyword."""
        strategy = _decide_strategy("Our insurance policy covers medical expenses.")
        assert strategy == "rag_augmented"

    def test_rag_augmented_compliance_keyword(self):
        """Should select RAG_AUGMENTED for compliance keyword."""
        strategy = _decide_strategy("We need to ensure compliance with regulations.")
        assert strategy == "rag_augmented"

    def test_direct_llm_no_special_keywords(self):
        """Should select DIRECT_LLM when no keywords match."""
        strategy = _decide_strategy("This is a simple statement about something.")
        assert strategy == "direct_llm"

    def test_rule_only_takes_precedence(self):
        """RULE_ONLY should take precedence over RAG keywords."""
        strategy = _decide_strategy(
            "Our investment guarantees 8% returns with insurance coverage."
        )
        # "guaranteed" should trigger RULE_ONLY even though investment keywords present
        assert strategy == "rule_only"


# =========================================================================
# Rule-Only Strategy Tests
# =========================================================================


class TestRunRules:
    """Test the _run_rules strategy."""

    def test_run_rules_returns_compliance_result(self, mock_llm_client: MagicMock):
        """_run_rules should return a ComplianceResult."""
        result = _run_rules(mock_llm_client, "Some transcript")
        assert isinstance(result, ComplianceResult)

    def test_run_rules_uses_fallback_engine(self, mock_llm_client: MagicMock):
        """_run_rules should call _fallback_rule_engine."""
        _run_rules(mock_llm_client, "Guaranteed returns!")
        mock_llm_client._fallback_rule_engine.assert_called_once()

    def test_run_rules_violation(self, mock_llm_client: MagicMock):
        """_run_rules should detect violations from rule engine."""
        mock_llm_client._fallback_rule_engine.return_value = {
            "violation": True,
            "confidence": 0.95,
            "reason": "Detected guaranteed returns.",
            "suggestion": "Remove guarantee language.",
            "risk_level": "high",
        }
        result = _run_rules(mock_llm_client, "Guaranteed 100% returns!")
        assert result.violation is True
        assert result.confidence == 0.95


# =========================================================================
# Direct LLM Strategy Tests
# =========================================================================


class TestRunLlm:
    """Test the _run_llm strategy."""

    def test_run_llm_returns_compliance_result(self, mock_llm_client: MagicMock):
        """_run_llm should return a ComplianceResult."""
        result = _run_llm(mock_llm_client, "Simple statement", "fintech")
        assert isinstance(result, ComplianceResult)

    def test_run_llm_calls_analyse(self, mock_llm_client: MagicMock):
        """_run_llm should call LLMClient.analyse()."""
        _run_llm(mock_llm_client, "Test transcript", "fintech")
        mock_llm_client.analyse.assert_called_once()

    def test_run_llm_uses_domain(self, mock_llm_client: MagicMock):
        """_run_llm should include domain in system prompt."""
        _run_llm(mock_llm_client, "Test", "healthcare")
        # Verify analyse was called with correct parameters
        mock_llm_client.analyse.assert_called_once()


# =========================================================================
# RAG Pipeline Tests
# =========================================================================


class TestRunRagPipeline:
    """Test the _run_rag_pipeline strategy."""

    def test_rag_pipeline_returns_compliance_result(
        self, mock_llm_client: MagicMock, mock_retriever: MagicMock
    ):
        """_run_rag_pipeline should return a ComplianceResult."""
        result = _run_rag_pipeline(
            mock_llm_client, mock_retriever, "Test transcript", "fintech"
        )
        assert isinstance(result, ComplianceResult)

    def test_rag_pipeline_calls_retrieve(
        self, mock_llm_client: MagicMock, mock_retriever: MagicMock
    ):
        """_run_rag_pipeline should call retriever.retrieve()."""
        _run_rag_pipeline(
            mock_llm_client, mock_retriever, "Test transcript", "fintech"
        )
        mock_retriever.retrieve.assert_called_once()

    def test_rag_pipeline_calls_analyse(
        self, mock_llm_client: MagicMock, mock_retriever: MagicMock
    ):
        """_run_rag_pipeline should call LLMClient.analyse() with context."""
        _run_rag_pipeline(
            mock_llm_client, mock_retriever, "Test transcript", "fintech"
        )
        mock_llm_client.analyse.assert_called_once()
        # Verify context was passed
        call_kwargs = mock_llm_client.analyse.call_args[1]
        assert "context" in call_kwargs
        assert call_kwargs["context"] is not None

    def test_rag_pipeline_blends_confidence(
        self, mock_llm_client: MagicMock, mock_retriever: MagicMock
    ):
        """_run_rag_pipeline should blend LLM and grounding scores."""
        mock_llm_client.analyse.return_value = LLMResponse(
            violation=False,
            confidence=0.8,
            reason="Test",
        )
        result = _run_rag_pipeline(
            mock_llm_client, mock_retriever, "Test transcript", "fintech"
        )
        # Confidence should be blended
        assert 0.0 <= result.confidence <= 1.0


# =========================================================================
# Helper Function Tests
# =========================================================================


class TestFormatRagContext:
    """Test _format_rag_context formatting."""

    def test_format_rag_context_empty_chunks(self):
        """Should handle empty chunk list gracefully."""
        result = _format_rag_context([])
        assert "No relevant policies" in result

    def test_format_rag_context_single_chunk(self):
        """Should format a single chunk."""
        chunk = PolicyChunk(
            source_file="test.txt",
            content="Test policy content.",
            domain="fintech",
        )
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.9,
            matched_keywords=["test"],
            query_keyword_count=1,
        )
        result = _format_rag_context([retrieved])
        assert "test.txt" in result
        assert "0.90" in result
        assert "RELEVANT REGULATORY POLICIES" in result

    def test_format_rag_context_multiple_chunks(self):
        """Should format multiple chunks with numbering."""
        chunk1 = PolicyChunk(source_file="policy1.txt", content="Content 1", domain="fintech")
        chunk2 = PolicyChunk(source_file="policy2.txt", content="Content 2", domain="fintech")
        retrieved_chunks = [
            RetrievedChunk(chunk=chunk1, score=0.9, matched_keywords=["test"], query_keyword_count=1),
            RetrievedChunk(chunk=chunk2, score=0.8, matched_keywords=["test"], query_keyword_count=1),
        ]
        result = _format_rag_context(retrieved_chunks)
        assert "Policy 1" in result
        assert "Policy 2" in result


class TestCalculateGroundingScore:
    """Test _calculate_grounding_score scoring."""

    def test_grounding_score_empty_chunks(self):
        """Should return 0.0 for empty chunks."""
        score = _calculate_grounding_score([])
        assert score == 0.0

    def test_grounding_score_single_chunk(self):
        """Should calculate score from single chunk."""
        chunk = PolicyChunk(source_file="test.txt", content="Test", domain="fintech")
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.8,
            matched_keywords=["test"],
            query_keyword_count=1,
        )
        score = _calculate_grounding_score([retrieved])
        # Single chunk + 5% boost = 0.85
        assert score > 0.8
        assert score <= 1.0

    def test_grounding_score_multiple_chunks(self):
        """Should boost score with multiple chunks."""
        chunks = []
        for i in range(3):
            chunk = PolicyChunk(source_file=f"test{i}.txt", content="Test", domain="fintech")
            retrieved = RetrievedChunk(
                chunk=chunk,
                score=0.8,
                matched_keywords=["test"],
                query_keyword_count=1,
            )
            chunks.append(retrieved)

        score = _calculate_grounding_score(chunks)
        # Average 0.8 + (3 * 0.05) boost = 0.8 + 0.15 = 0.95
        assert 0.9 < score <= 1.0


# =========================================================================
# ComplianceAgent Tests
# =========================================================================


class TestComplianceAgent:
    """Test the main ComplianceAgent class."""

    def test_agent_initialization(self, mock_llm_client: MagicMock, mock_retriever: MagicMock):
        """Agent should initialize with dependencies."""
        agent = ComplianceAgent(mock_llm_client, mock_retriever)
        assert agent.llm_client is mock_llm_client
        assert agent.retriever is mock_retriever

    def test_agent_analyse_validates_transcript(
        self, mock_llm_client: MagicMock, mock_retriever: MagicMock
    ):
        """Agent.analyse() should validate non-empty transcript."""
        agent = ComplianceAgent(mock_llm_client, mock_retriever)
        with pytest.raises(ValueError, match="Transcript cannot be empty"):
            agent.analyse("", "fintech")

    def test_agent_analyse_validates_domain(
        self, mock_llm_client: MagicMock, mock_retriever: MagicMock
    ):
        """Agent.analyse() should validate non-empty domain."""
        agent = ComplianceAgent(mock_llm_client, mock_retriever)
        with pytest.raises(ValueError, match="Domain cannot be empty"):
            agent.analyse("Test transcript", "")

    def test_agent_analyse_rule_only_strategy(self, agent: ComplianceAgent):
        """Agent should execute RULE_ONLY strategy."""
        result = agent.analyse("We guarantee 100% returns!", "fintech")
        assert isinstance(result, ComplianceResult)
        # Rule-only strategies typically set high confidence
        assert result.confidence > 0.5

    def test_agent_analyse_direct_llm_strategy(self, agent: ComplianceAgent):
        """Agent should execute DIRECT_LLM strategy."""
        result = agent.analyse("This is a simple statement.", "fintech")
        assert isinstance(result, ComplianceResult)

    def test_agent_analyse_rag_augmented_strategy(self, agent: ComplianceAgent):
        """Agent should execute RAG_AUGMENTED strategy."""
        result = agent.analyse("Tell me about investment returns.", "fintech")
        assert isinstance(result, ComplianceResult)

    def test_agent_analyse_no_retriever_fallback(self, mock_llm_client: MagicMock):
        """Agent should fall back to DIRECT_LLM without retriever."""
        agent = ComplianceAgent(mock_llm_client, retriever=None)
        result = agent.analyse("Talk about investment policy.", "fintech")
        assert isinstance(result, ComplianceResult)

    def test_agent_analyse_returns_consistent_schema(self, agent: ComplianceAgent):
        """Agent should always return ComplianceResult with consistent schema."""
        transcripts = [
            "We guarantee returns!",
            "Simple statement.",
            "Investment policy details.",
        ]
        for transcript in transcripts:
            result = agent.analyse(transcript, "fintech")
            assert isinstance(result, ComplianceResult)
            assert hasattr(result, "violation")
            assert hasattr(result, "risk_level")
            assert hasattr(result, "confidence")
            assert hasattr(result, "reason")
            assert hasattr(result, "suggestion")

    def test_agent_analyse_error_handling(self, mock_llm_client: MagicMock):
        """Agent should handle errors with fallback."""
        mock_llm_client.analyse.side_effect = Exception("LLM error")
        mock_llm_client._fallback_rule_engine.return_value = {
            "violation": False,
            "confidence": 0.5,
            "reason": "Fallback mode",
            "suggestion": "Manual review needed",
            "risk_level": "low",
        }

        agent = ComplianceAgent(mock_llm_client, retriever=None, use_fallback_on_error=True)
        result = agent.analyse("Test transcript", "fintech")
        assert isinstance(result, ComplianceResult)

    def test_agent_analyse_error_raise_when_fallback_disabled(
        self, mock_llm_client: MagicMock
    ):
        """Agent should raise error when fallback is disabled."""
        mock_llm_client.analyse.side_effect = Exception("LLM error")

        agent = ComplianceAgent(mock_llm_client, retriever=None, use_fallback_on_error=False)
        with pytest.raises(Exception):
            agent.analyse("Test transcript", "fintech")
