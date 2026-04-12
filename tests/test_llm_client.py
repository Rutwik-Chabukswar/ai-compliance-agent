"""
tests/test_llm_client.py
------------------------
Unit tests for the offline local LLM client.
"""

import json

import pytest

from compliance_engine.llm_client import LLMClient, LLMClientError, LLMResponse


class TestLLMClient:
    def test_analyse_detects_high_risk_phrases(self):
        client = LLMClient()

        response = client.analyse("We guarantee a zero risk investment opportunity.")
        assert response.violation is True
        assert response.confidence == 0.9
        assert "high-risk" in response.reason

    def test_analyse_detects_medium_risk_phrases(self):
        client = LLMClient()

        response = client.analyse("This is a high returns exclusive deal.")
        assert response.violation is True
        assert response.confidence == 0.6
        assert "medium-risk" in response.reason

    def test_analyse_non_violation(self):
        client = LLMClient()

        response = client.analyse("This product description is factual and balanced.")
        assert response.violation is False
        assert response.confidence == 0.1
        assert "No explicit high-risk" in response.reason

    def test_analyse_context_increases_confidence(self):
        client = LLMClient()
        context = "Guaranteed returns are prohibited by securities law."

        response = client.analyse(
            "We guarantee a 15% return.",
            context=context,
        )
        assert response.violation is True
        assert response.confidence == 1.0 or response.confidence == 0.9
        assert "Matching policy context increased confidence" in response.reason

    def test_chat_returns_json_compatible_output(self):
        client = LLMClient()

        payload = client.chat("system prompt", "We guarantee returns", system_context="policy text")
        result = json.loads(payload)

        assert result["violation"] is True
        assert isinstance(result["confidence"], float)
        assert "reason" in result
