"""
config.py
---------
Central configuration for the AI Compliance Detection Engine.
All tuneable parameters live here so they can be overridden via
environment variables without touching source code.
"""

import os

# ---------------------------------------------------------------------------
# LLM API settings
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Default model – override with env var to switch between OSS / proprietary
DEFAULT_MODEL: str = os.getenv("COMPLIANCE_MODEL", "gpt-4o-mini")

# Inference hyper-parameters
TEMPERATURE: float = float(os.getenv("COMPLIANCE_TEMPERATURE", "0.0"))  # deterministic
MAX_TOKENS: int = int(os.getenv("COMPLIANCE_MAX_TOKENS", "512"))

# ---------------------------------------------------------------------------
# Retry / timeout settings
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("COMPLIANCE_TIMEOUT", "30"))
MAX_RETRIES: int = int(os.getenv("COMPLIANCE_MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR: float = float(os.getenv("COMPLIANCE_BACKOFF", "1.5"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("COMPLIANCE_LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation) settings
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL: str = os.getenv("COMPLIANCE_EMBEDDING_MODEL", "text-embedding-3-small")
POLICIES_DIR: str = os.getenv("COMPLIANCE_POLICIES_DIR", "configs/policies")
RAG_TOP_K: int = int(os.getenv("COMPLIANCE_RAG_TOP_K", "3"))
