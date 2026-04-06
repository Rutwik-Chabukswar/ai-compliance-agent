"""
llm_client.py
-------------
Thin, opinionated wrapper around the OpenAI Chat Completions API.

Design goals
~~~~~~~~~~~~
* Single responsibility: send a chat prompt → receive a text response.
* Configurable model so we can point at any OpenAI-compatible endpoint
  (e.g., locally-hosted OSS models via vLLM / LM Studio).
* Graceful retry with exponential back-off for transient 5xx / network errors.
* Hard timeout so a stalled request never blocks the caller indefinitely.
* Structured logging at every step for production observability.

Future RAG integration point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``LLMClient.chat()`` accepts an optional ``system_context`` kwarg that is
prepended to the system prompt. A RAG retrieval step can populate this with
domain-specific policy chunks before calling the engine.
"""

import logging
import time
from typing import Optional

import openai
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

from compliance_engine.config import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    MAX_TOKENS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_FACTOR,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised when the LLM client cannot produce a response after all retries."""


class LLMClient:
    """
    Wrapper around ``openai.OpenAI`` that adds retry logic and timeout control.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``,
        ``"neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"``).
    api_key:
        OpenAI API key. Defaults to ``config.OPENAI_API_KEY``.
    base_url:
        API base URL. Override to point at a local vLLM / Ollama endpoint.
    temperature:
        Sampling temperature. Keep at 0.0 for deterministic compliance output.
    max_tokens:
        Upper bound on response tokens.
    max_retries:
        Number of retry attempts on transient failures.
    timeout:
        Per-request timeout in seconds.
    retry_backoff:
        Multiplicative back-off factor between successive retries.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = MAX_RETRIES,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        retry_backoff: float = RETRY_BACKOFF_FACTOR,
    ) -> None:
        if not api_key:
            raise ValueError(
                "No OpenAI API key provided. "
                "Set OPENAI_API_KEY in your environment or pass api_key= explicitly."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_backoff = retry_backoff

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout),
            max_retries=0,  # We handle retries ourselves for full observability.
        )

        logger.info(
            "LLMClient initialised | model=%s | temperature=%s | timeout=%ds",
            self.model,
            self.temperature,
            self.timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        system_context: Optional[str] = None,
    ) -> str:
        """
        Send a chat-completion request and return the assistant's text reply.

        Parameters
        ----------
        system_prompt:
            The base system instruction (role definition + output format).
        user_prompt:
            The user turn containing the transcript and domain.
        system_context:
            Optional extra context injected at the top of the system prompt
            (intended for RAG-retrieved policy chunks in future iterations).

        Returns
        -------
        str
            Raw text content of the first choice's message.

        Raises
        ------
        LLMClientError
            If all retry attempts are exhausted without a successful response.
        """
        effective_system = (
            f"{system_context}\n\n{system_prompt}" if system_context else system_prompt
        )

        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": user_prompt},
        ]

        attempt = 0
        last_error: Exception = RuntimeError("No attempts made")

        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.debug(
                    "LLM request attempt %d/%d | model=%s",
                    attempt,
                    self.max_retries,
                    self.model,
                )
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},  # enforce JSON mode
                )
                content = response.choices[0].message.content or ""
                logger.debug(
                    "LLM response received | tokens_used=%s | preview=%s",
                    response.usage.total_tokens if response.usage else "?",
                    content[:120],
                )
                return content

            except RateLimitError as exc:
                logger.warning(
                    "Rate limit hit on attempt %d/%d – backing off.",
                    attempt,
                    self.max_retries,
                )
                last_error = exc

            except APIConnectionError as exc:
                logger.warning(
                    "Connection error on attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                last_error = exc

            except APIStatusError as exc:
                # 5xx → retryable; 4xx (except 429) → fatal
                if exc.status_code >= 500:
                    logger.warning(
                        "Server error %d on attempt %d/%d: %s",
                        exc.status_code,
                        attempt,
                        self.max_retries,
                        exc.message,
                    )
                    last_error = exc
                else:
                    logger.error(
                        "Non-retryable API error %d: %s",
                        exc.status_code,
                        exc.message,
                    )
                    raise LLMClientError(
                        f"Non-retryable API error {exc.status_code}: {exc.message}"
                    ) from exc

            # Exponential back-off (skip sleep after the last attempt)
            if attempt < self.max_retries:
                sleep_time = self.retry_backoff ** attempt
                logger.debug("Sleeping %.2fs before next retry.", sleep_time)
                time.sleep(sleep_time)

        raise LLMClientError(
            f"LLM request failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error
