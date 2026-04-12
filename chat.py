"""
chat.py
-------
Interactive terminal chat mode for the AI compliance detection system.
"""

import logging
import sys

from compliance_engine.agent import ComplianceAgent
from compliance_engine.llm_client import LLMClient
from compliance_engine.rag import PolicyRetriever, load_policies_from_directory
from compliance_engine.config import POLICIES_DIR


def setup_minimized_logging():
    """Keep logs minimal in chat mode to avoid cluttering UX."""
    # Suppress output from backend components to keep the chat interface clean
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("compliance_engine").setLevel(logging.ERROR)
    
    # Configure root logger
    logging.basicConfig(level=logging.ERROR, format="%(message)s")


def init_agent() -> ComplianceAgent:
    """Initialize the backend components: LLM, RAG, and Agent orchestration."""
    try:
        llm_client = LLMClient()
    except Exception as e:
        print(f"Failed to initialize LLM Client: {e}")
        sys.exit(1)

    # Attempt to load RAG policies safely
    try:
        policies = load_policies_from_directory(POLICIES_DIR)
        retriever = PolicyRetriever(policies) if policies else None
    except Exception:
        # Failsafe if policies folder is missing
        retriever = None

    return ComplianceAgent(llm_client=llm_client, retriever=retriever)


def run_chat():
    """Main conversational loop for real-time compliance feedback."""
    setup_minimized_logging()

    print("Initializing compliance agent... Please wait.")
    agent = init_agent()

    print("=" * 60)
    print(" AI Compliance Chat Mode (type 'exit' to quit)")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        print("Agent:")
        
        try:
            # Evaluate using the backend compliance engine
            result = agent.analyse(user_input, domain="fintech")

            if result.violation:
                print("⚠️ Violation: YES")
                print(f"Risk: {result.risk_level.upper()}")
                print(f"Reason: {result.reason}")
                print(f"Suggestion: {result.suggestion}")
            else:
                print("✅ No compliance issues detected")
        except Exception as e:
            print(f"⚠️ Error evaluating input: {e}")
        
        print()


if __name__ == "__main__":
    run_chat()
