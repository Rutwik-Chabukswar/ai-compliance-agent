"""
tests/test_rag_free_mode.py
---------------------------
Unit tests for FREE MODE RAG system (no embeddings API calls).

Tests verify that:
- Embeddings client operates in FREE MODE (returns dummy vectors)
- Retriever uses keyword-based matching instead of vector similarity
- All policy chunks can be retrieved without external API calls
- Domain filtering works correctly in FREE MODE
- Integration with ComplianceEngine works end-to-end

Run with:
    pytest -v tests/test_rag_free_mode.py
"""

import pytest
from unittest.mock import MagicMock

from compliance_engine.rag.embeddings import EmbeddingsClient
from compliance_engine.rag.retriever import (
    PolicyRetriever,
    _extract_keywords,
    _keyword_similarity,
)
from compliance_engine.rag.loader import PolicyChunk
from compliance_engine.compliance_engine import ComplianceEngine
from compliance_engine.llm_client import LLMClient


class TestEmbeddingsClientFreeMode:
    """Test EmbeddingsClient in FREE MODE."""

    def test_free_mode_auto_detect_no_api_key(self):
        """FREE MODE should auto-enable when API key is empty."""
        client = EmbeddingsClient(api_key="")
        assert client.use_free_mode is True

    def test_free_mode_explicit(self):
        """FREE MODE can be explicitly enabled."""
        client = EmbeddingsClient(use_free_mode=True)
        assert client.use_free_mode is True

    def test_free_mode_returns_dummy_vector(self):
        """Embeddings should return [0.0] in FREE MODE."""
        client = EmbeddingsClient(use_free_mode=True)
        vec = client.embed_text("Some text to embed")
        assert vec == [0.0]

    def test_free_mode_no_api_call(self):
        """FREE MODE should not attempt to call OpenAI API."""
        client = EmbeddingsClient(use_free_mode=True, api_key="")
        # This should not raise an error or attempt API call
        vec = client.embed_text("Test text")
        assert vec == [0.0]

    def test_free_mode_empty_text(self):
        """Embeddings should handle empty text in FREE MODE."""
        client = EmbeddingsClient(use_free_mode=True)
        vec = client.embed_text("")
        assert vec == [0.0]


class TestKeywordExtraction:
    """Test keyword extraction for FREE MODE matching."""

    def test_extract_keywords_basic(self):
        """Extract keywords from simple text."""
        keywords = _extract_keywords("We guarantee 15% returns with zero risk")
        assert "guarantee" in keywords
        assert "returns" in keywords
        assert "risk" in keywords

    def test_extract_keywords_filters_stop_words(self):
        """Stop words should be filtered out."""
        keywords = _extract_keywords("the and or a an is are")
        assert len(keywords) == 0  # All are stop words

    def test_extract_keywords_minimum_length(self):
        """Keywords shorter than min_length should be filtered."""
        keywords = _extract_keywords("I am going to test this", min_length=3)
        # "I" and "am" and "to" are < 3 chars
        assert "test" in keywords
        assert "going" in keywords

    def test_extract_keywords_lowercase(self):
        """Keywords should be lowercase."""
        keywords = _extract_keywords("GUARANTEED RETURNS Risk")
        assert "guaranteed" in keywords
        assert "returns" in keywords
        assert "risk" in keywords


class TestKeywordSimilarity:
    """Test keyword-based similarity scoring."""

    def test_similarity_perfect_match(self):
        """Score should equal the raw overlap count for perfect keyword match."""
        query = "fraud compliance review"
        chunk = "This document discusses fraud and compliance and review procedures"
        score = _keyword_similarity(query, chunk)
        assert score == 3.0

    def test_similarity_no_match(self):
        """Score should be 0.0 for no keyword overlap."""
        query = "investment returns"
        chunk = "This is about food preparation and cooking techniques"
        score = _keyword_similarity(query, chunk)
        assert score == 0.0

    def test_similarity_partial_match(self):
        """Score should reflect the raw number of matching keywords."""
        query = "guaranteed returns"
        chunk = "Guaranteed returns are illegal"
        score = _keyword_similarity(query, chunk)
        assert score == 2.0

    def test_similarity_empty_query(self):
        """Empty query should return 0.0 score."""
        score = _keyword_similarity("", "Some chunk content")
        assert score == 0.0


class TestPolicyRetrieverFreeMode:
    """Test PolicyRetriever with FREE MODE."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample policy chunks for testing."""
        return [
            PolicyChunk(
                source_file="securities_law.txt",
                content="Guaranteed returns promise in investment marketing is illegal. "
                        "Securities law prohibits guaranteeing returns or protecting principal.",
                domain="fintech",
            ),
            PolicyChunk(
                source_file="insurance_rules.txt",
                content="Insurance policies must clearly disclose coverage limitations. "
                        "Premium and deductible information is mandatory.",
                domain="insurance",
            ),
            PolicyChunk(
                source_file="fintech_policy.txt",
                content="Financial technology services must comply with KYC/AML regulations. "
                        "Customer verification is required before trading.",
                domain="fintech",
            ),
        ]

    def test_retriever_free_mode(self, sample_chunks):
        """Retriever should use FREE MODE automatically."""
        client = EmbeddingsClient(api_key="")
        retriever = PolicyRetriever(sample_chunks, embeddings_client=client)
        assert retriever.client.use_free_mode is True

    def test_retriever_no_vector_index_in_free_mode(self, sample_chunks):
        """Vector index should be empty in FREE MODE."""
        client = EmbeddingsClient(use_free_mode=True)
        retriever = PolicyRetriever(sample_chunks, embeddings_client=client)
        # In FREE MODE, vectors list should remain empty
        assert len(retriever.vectors) == 0

    def test_retriever_retrieve_top_k(self, sample_chunks):
        """Retrieve should return top_k results."""
        client = EmbeddingsClient(use_free_mode=True)
        retriever = PolicyRetriever(sample_chunks, embeddings_client=client)
        
        results = retriever.retrieve(
            query="guaranteed returns in investment",
            top_k=2,
            domain="fintech",
        )
        
        assert len(results) <= 2
        assert all(isinstance(chunk, PolicyChunk) for chunk, score in results)
        assert all(isinstance(score, float) for chunk, score in results)

    def test_retriever_domain_filtering(self, sample_chunks):
        """Retriever should filter by domain correctly."""
        client = EmbeddingsClient(use_free_mode=True)
        retriever = PolicyRetriever(sample_chunks, embeddings_client=client)
        
        # Query for fintech only
        fintech_results = retriever.retrieve(
            query="returns investment",
            top_k=10,
            domain="fintech",
        )
        
        # Query for insurance only
        insurance_results = retriever.retrieve(
            query="coverage premium",
            top_k=10,
            domain="insurance",
        )
        
        # Verify correct filtering
        assert all(chunk.domain == "fintech" for chunk, _ in fintech_results)
        assert all(chunk.domain == "insurance" for chunk, _ in insurance_results)

    def test_retriever_scoring_order(self, sample_chunks):
        """Results should be sorted by score (descending)."""
        client = EmbeddingsClient(use_free_mode=True)
        retriever = PolicyRetriever(sample_chunks, embeddings_client=client)
        
        results = retriever.retrieve(
            query="guaranteed returns illegal",
            top_k=10,
        )
        
        # Scores should be in descending order
        scores = [score for chunk, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_retriever_empty_chunks(self):
        """Retriever with no chunks should return empty list."""
        client = EmbeddingsClient(use_free_mode=True)
        retriever = PolicyRetriever([], embeddings_client=client)
        
        results = retriever.retrieve("test query", top_k=5)
        assert results == []


class TestComplianceEngineWithFreeMode:
    """Integration tests for ComplianceEngine with FREE MODE RAG."""

    def test_compliance_engine_with_free_mode_retriever(self):
        """ComplianceEngine should work with FREE MODE retriever."""
        # Create chunks
        chunks = [
            PolicyChunk(
                source_file="test.txt",
                content="Guaranteed returns are prohibited",
                domain="fintech",
            ),
        ]
        
        # Create FREE MODE retriever
        client = EmbeddingsClient(use_free_mode=True)
        retriever = PolicyRetriever(chunks, embeddings_client=client)
        
        # Create mock LLM client
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = (
            '{"violation": true, "risk_level": "high", "confidence": 0.95, '
            '"reason": "Guaranteed returns claim detected", '
            '"suggestion": "Remove guarantee language"}'
        )
        
        # Create engine with FREE MODE retriever
        engine = ComplianceEngine(
            llm_client=mock_llm,
            retriever=retriever,
            use_fallback_on_error=False,
        )
        
        # Analyze (should work without API calls for embeddings)
        result = engine.analyse(
            transcript="We guarantee 15% returns",
            domain="fintech",
        )
        
        assert result.violation is True
        assert result.risk_level == "high"

    def test_compliance_engine_retrieval_used_in_analysis(self):
        """RAG context should be included in LLM prompt."""
        chunks = [
            PolicyChunk(
                source_file="securities.txt",
                content="Guaranteed returns are illegal under securities law",
                domain="fintech",
            ),
        ]
        
        client = EmbeddingsClient(use_free_mode=True)
        retriever = PolicyRetriever(chunks, embeddings_client=client)
        
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = (
            '{"violation": true, "risk_level": "high", "confidence": 0.95, '
            '"reason": "Matches policy", '
            '"suggestion": "Fix"}'
        )
        
        engine = ComplianceEngine(
            llm_client=mock_llm,
            retriever=retriever,
            use_fallback_on_error=False,
        )
        
        engine.analyse(
            transcript="We guarantee returns",
            domain="fintech",
        )
        
        # Verify RAG context was included in the system prompt
        call_args = mock_llm.chat.call_args
        system_prompt = call_args.kwargs.get("system_prompt") or call_args[0][0]
        
        # Should contain RAG context with retrieved policies
        assert "RELEVANT REGULATORY POLICIES" in system_prompt or "securities" in system_prompt.lower()
