"""
tests/test_llm_client.py
------------------------
Unit tests for the LLM client (retry/timeout logic mocked).
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import openai

from compliance_engine.llm_client import LLMClient, LLMClientError


@pytest.fixture()
def client():
    with patch("compliance_engine.llm_client.OpenAI"):
        return LLMClient(api_key="test-key", max_retries=3, retry_backoff=0.0)


class TestLLMClient:
    def test_no_api_key_raises(self):
        with pytest.raises(ValueError, match="No OpenAI API key"):
            LLMClient(api_key="")

    def test_successful_response(self, client):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"violation": false}'
        mock_response.usage.total_tokens = 42
        client._client.chat.completions.create.return_value = mock_response

        result = client.chat("system", "user")
        assert result == '{"violation": false}'

    def test_retries_on_rate_limit(self, client):
        """Should retry on RateLimitError and eventually succeed."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"violation": false}'
        mock_response.usage.total_tokens = 10

        client._client.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit", response=MagicMock(), body={}),
            openai.RateLimitError("Rate limit", response=MagicMock(), body={}),
            mock_response,
        ]

        with patch("time.sleep"):
            result = client.chat("system", "user")
        assert result == '{"violation": false}'
        assert client._client.chat.completions.create.call_count == 3

    def test_raises_after_max_retries(self, client):
        """Should raise LLMClientError after exhausting all retries."""
        client._client.chat.completions.create.side_effect = (
            openai.RateLimitError("Rate limit", response=MagicMock(), body={})
        )
        with patch("time.sleep"), pytest.raises(LLMClientError, match="failed after"):
            client.chat("system", "user")

    def test_non_retryable_4xx_raises_immediately(self, client):
        """4xx errors (except 429) should raise immediately without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        client._client.chat.completions.create.side_effect = openai.BadRequestError(
            "Bad request", response=mock_response, body={}
        )
        with pytest.raises(LLMClientError, match="Non-retryable"):
            client.chat("system", "user")
        assert client._client.chat.completions.create.call_count == 1

    def test_system_context_prepended(self, client):
        """system_context should be prepended to the system prompt."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "{}"
        mock_response.usage.total_tokens = 5
        client._client.chat.completions.create.return_value = mock_response

        client.chat("Base system.", "User msg.", system_context="RAG chunk here.")

        call_kwargs = client._client.chat.completions.create.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert system_msg.startswith("RAG chunk here.")
        assert "Base system." in system_msg
