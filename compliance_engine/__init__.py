"""
compliance_engine – AI-powered compliance detection engine.

Public surface
--------------
    from compliance_engine import ComplianceAgent, ComplianceResult

    agent = ComplianceAgent(llm_client=..., retriever=...)
    result = agent.analyse(transcript="...", domain="fintech")
    print(result.to_json())
"""

import logging

from compliance_engine.agent import ComplianceAgent
from compliance_engine.compliance_engine import ComplianceEngine, ComplianceResult, DebugInfo
from compliance_engine.llm_client import LLMClient
from compliance_engine.config import LOG_LEVEL

__all__ = [
    "ComplianceAgent",
    "ComplianceEngine",
    "ComplianceResult",
    "DebugInfo",
    "LLMClient",
]

# Configure package-level logger.
# Applications can override this by configuring the root logger before import.
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
