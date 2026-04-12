"""
example_agent_usage.py
----------------------
Example usage of the ComplianceAgent orchestration layer.

This demonstrates how to:
  1. Initialize the agent with LLM and RAG components
  2. Analyze transcripts using dynamic strategy selection
  3. Handle different compliance scenarios
"""

import json
from compliance_engine import ComplianceAgent
from compliance_engine.llm_client import LLMClient
from compliance_engine.rag import PolicyRetriever, load_policies_from_directory
from compliance_engine.config import POLICIES_DIR


def example_basic_usage():
    """Basic agent usage with LLM only (no RAG)."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Agent Usage (LLM Only)")
    print("=" * 70)

    # Initialize components
    llm_client = LLMClient()
    agent = ComplianceAgent(llm_client=llm_client, retriever=None)

    # Analyze a simple transcript
    transcript = "This investment historically returned 7% annually with no guarantees."
    result = agent.analyse(transcript, domain="fintech")

    print(f"\nTranscript: {transcript}")
    print(f"Result:\n{result.to_json(indent=2)}")
    print()


def example_rule_only_detection():
    """Agent automatically selects RULE_ONLY strategy for obvious violations."""
    print("=" * 70)
    print("EXAMPLE 2: Automatic RULE_ONLY Strategy Detection")
    print("=" * 70)

    llm_client = LLMClient()
    agent = ComplianceAgent(llm_client=llm_client, retriever=None)

    # Transcript with red-flag keywords triggers rule-only strategy
    transcript = "We guarantee 100% returns with zero risk!"
    result = agent.analyse(transcript, domain="fintech")

    print(f"\nTranscript: {transcript}")
    print(f"Strategy: RULE_ONLY (detected by keyword matching)")
    print(f"Result:\n{result.to_json(indent=2)}")
    print()


def example_rag_augmented_analysis():
    """Agent uses RAG-augmented strategy with policy context."""
    print("=" * 70)
    print("EXAMPLE 3: RAG-Augmented Strategy with Policy Context")
    print("=" * 70)

    # Initialize LLM and RAG
    llm_client = LLMClient()
    
    # Load policies from directory (or create dummy if not available)
    try:
        policies = load_policies_from_directory(POLICIES_DIR)
    except Exception:
        print("Note: Policy directory not available. Using LLM-only mode.")
        policies = []
    
    retriever = PolicyRetriever(policies) if policies else None
    agent = ComplianceAgent(llm_client=llm_client, retriever=retriever)

    # Transcript with domain-specific keywords triggers RAG strategy
    transcript = (
        "Our investment policy offers returns based on market performance. "
        "Insurance coverage includes medical and disability benefits."
    )
    result = agent.analyse(transcript, domain="fintech")

    print(f"\nTranscript: {transcript[:80]}...")
    print(f"Strategy: RAG_AUGMENTED (domain keywords detected)")
    if retriever and result.debug:
        print(f"Retrieved Policies: {len(result.debug.get('retrieved_policies', []))}")
    print(f"Result:\n{result.to_json(indent=2)}")
    print()


def example_strategy_comparison():
    """Compare how different transcripts are handled."""
    print("=" * 70)
    print("EXAMPLE 4: Strategy Comparison Across Transcripts")
    print("=" * 70)

    llm_client = LLMClient()
    agent = ComplianceAgent(llm_client=llm_client, retriever=None)

    test_cases = [
        ("We guarantee 200% annual returns!", "RULE_ONLY"),
        ("Past performance is not indicative of future results.", "DIRECT_LLM"),
        ("Our policy covers investment returns and insurance claims.", "RAG_AUGMENTED"),
        ("This product is risk-free and guaranteed.", "RULE_ONLY"),
    ]

    for transcript, expected_strategy in test_cases:
        try:
            result = agent.analyse(transcript, domain="fintech")
            print(f"\nTranscript: {transcript}")
            print(f"Expected Strategy: {expected_strategy}")
            print(f"Violation: {result.violation} | Risk: {result.risk_level} | Confidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"\nTranscript: {transcript}")
            print(f"Error: {str(e)}")


def example_compliance_workflow():
    """Demonstrate a typical compliance workflow."""
    print("=" * 70)
    print("EXAMPLE 5: Compliance Detection Workflow")
    print("=" * 70)

    llm_client = LLMClient()
    agent = ComplianceAgent(llm_client=llm_client, retriever=None)

    # Marketing transcript for analysis
    marketing_transcript = """
    Welcome to our investment advisory service. We've been managing portfolios
    for 20 years. Our fund has historically achieved returns around 8% annually,
    though past performance is not a guarantee of future results. We recommend
    working with a financial advisor to understand the risks involved in any
    investment. Diversification and regular monitoring are key to success.
    """

    print(f"\nAnalyzing marketing transcript...")
    result = agent.analyse(marketing_transcript, domain="fintech")

    print(f"\nCompliance Analysis Results:")
    print(f"  Violation Detected: {result.violation}")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Reason: {result.reason}")
    print(f"  Suggestion: {result.suggestion}")

    if result.violation:
        print("\n⚠️  COMPLIANCE ISSUE DETECTED - Manual review recommended")
    else:
        print("\n✓ No compliance violations detected")


def example_error_handling():
    """Demonstrate error handling and fallback behavior."""
    print("=" * 70)
    print("EXAMPLE 6: Error Handling & Fallback Behavior")
    print("=" * 70)

    # Agent with fallback enabled (default)
    llm_client = LLMClient()
    agent_with_fallback = ComplianceAgent(
        llm_client=llm_client,
        retriever=None,
        use_fallback_on_error=True,
    )

    # Valid transcript should work
    transcript = "Our product offers market-linked returns."
    
    try:
        result = agent_with_fallback.analyse(transcript, domain="fintech")
        print(f"\n✓ Analysis succeeded")
        print(f"  Result: {result.violation} | Confidence: {result.confidence:.2f}")
    except Exception as e:
        print(f"\n✗ Analysis failed: {str(e)}")

    # Invalid inputs should raise errors
    print(f"\nTesting input validation:")
    
    try:
        agent_with_fallback.analyse("", "fintech")
    except ValueError as e:
        print(f"  ✓ Empty transcript rejected: {e}")
    
    try:
        agent_with_fallback.analyse("Valid transcript", "")
    except ValueError as e:
        print(f"  ✓ Empty domain rejected: {e}")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_rule_only_detection()
    example_rag_augmented_analysis()
    example_strategy_comparison()
    example_compliance_workflow()
    example_error_handling()

    print("\n" + "=" * 70)
    print("All examples completed")
    print("=" * 70)
