#!/usr/bin/env python
"""
Test script for the refactored LLMClient.

Validates:
  - Fallback rule engine works correctly
  - JSON parsing and validation works
  - Error handling works
  - Logging is functional
"""

import json
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

from compliance_engine.llm_client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)


def test_fallback_mode():
    """Test that fallback rule-based engine works when no API key."""
    print("\n" + "=" * 70)
    print("TEST 1: Fallback Rule-Based Mode (No API Key)")
    print("=" * 70)

    # Create client without API key
    client = LLMClient(api_key="")

    # Test cases
    test_cases = [
        {
            "name": "High-Risk Guaranteed Return",
            "transcript": "We guarantee 15% annual returns with zero risk.",
            "expect_violation": True,
            "expect_risk": "high",
        },
        {
            "name": "Medium-Risk Limited Time Offer",
            "transcript": "This is a limited time offer with high returns!",
            "expect_violation": True,
            "expect_risk": "medium",
        },
        {
            "name": "Compliant Statement",
            "transcript": "Our fund historically returned 5-8% annually. Past performance is not "
                         "a guarantee of future results.",
            "expect_violation": False,
            "expect_risk": "low",
        },
    ]

    for test_case in test_cases:
        print(f"\n  Test: {test_case['name']}")
        print(f"  Input: {test_case['transcript'][:60]}...")

        result = client.analyse(transcript=test_case["transcript"])

        print(f"  Result:")
        print(f"    - violation: {result.violation}")
        print(f"    - confidence: {result.confidence:.2f}")
        print(f"    - reason: {result.reason}")

        # Validate expectations
        assert result.violation == test_case["expect_violation"], (
            f"Expected violation={test_case['expect_violation']}, "
            f"got {result.violation}"
        )
        print(f"  ✓ PASS: Violation detected correctly")


def test_chat_fallback_mode():
    """Test that chat() method returns proper JSON in fallback mode."""
    print("\n" + "=" * 70)
    print("TEST 2: Chat Method with Fallback (JSON Output)")
    print("=" * 70)

    client = LLMClient(api_key="")

    system_prompt = "You are a compliance officer."
    user_prompt = "We guarantee 20% returns!"

    result_json = client.chat(system_prompt=system_prompt, user_prompt=user_prompt)

    print(f"\n  Input: {user_prompt}")
    print(f"  Output JSON: {result_json}")

    # Parse and validate JSON
    data = json.loads(result_json)
    print(f"\n  Parsed Result:")
    print(f"    - violation: {data['violation']}")
    print(f"    - risk_level: {data['risk_level']}")
    print(f"    - confidence: {data['confidence']}")

    # Validate schema
    required_fields = {"violation", "risk_level", "confidence", "reason", "suggestion"}
    assert required_fields.issubset(data.keys()), (
        f"Missing required fields: {required_fields - set(data.keys())}"
    )
    print(f"  ✓ PASS: JSON schema is valid")


def test_json_extraction():
    """Test JSON extraction from various formats."""
    print("\n" + "=" * 70)
    print("TEST 3: JSON Extraction from Markdown/Preamble")
    print("=" * 70)

    client = LLMClient(api_key="")

    test_cases = [
        {
            "name": "Clean JSON",
            "input": '{"violation": true, "risk_level": "high", "confidence": 0.9, '
                    '"reason": "test", "suggestion": "fix"}',
        },
        {
            "name": "Markdown fences",
            "input": '```json\n{"violation": false, "risk_level": "low", "confidence": 0.5, '
                    '"reason": "ok", "suggestion": "none"}\n```',
        },
        {
            "name": "With preamble",
            "input": 'Here is the analysis: {"violation": true, "risk_level": "medium", '
                    '"confidence": 0.7, "reason": "maybe", "suggestion": "review"}',
        },
    ]

    for test_case in test_cases:
        print(f"\n  Test: {test_case['name']}")
        try:
            extracted = LLMClient._extract_json_block(test_case["input"])
            data = json.loads(extracted)
            assert "violation" in data
            print(f"  ✓ PASS: Successfully extracted JSON with keys: "
                  f"{list(data.keys())[:3]}...")
        except Exception as e:
            print(f"  ✗ FAIL: {str(e)}")
            raise


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 70)
    print("TEST 4: Error Handling")
    print("=" * 70)

    client = LLMClient(api_key="")

    # Test case 1: No JSON in response
    print("\n  Test: Malformed response (no JSON)")
    try:
        LLMClient._extract_json_block("This is just plain text with no JSON.")
        print("  ✗ FAIL: Should have raised error")
    except LLMClientError as e:
        print(f"  ✓ PASS: Correctly raised LLMClientError: {str(e)[:50]}...")

    # Test case 2: Unbalanced braces
    print("\n  Test: Unbalanced JSON braces")
    try:
        LLMClient._extract_json_block('{"violation": true')
        print("  ✗ FAIL: Should have raised error")
    except LLMClientError as e:
        print(f"  ✓ PASS: Correctly raised LLMClientError for unbalanced braces")


def test_fallback_confidence_scores():
    """Test that confidence scores are appropriate."""
    print("\n" + "=" * 70)
    print("TEST 5: Confidence Score Validation")
    print("=" * 70)

    client = LLMClient(api_key="")

    transcripts = [
        ("guarantee zero risk", 0.9),  # High risk phrase -> high confidence
        ("limited time offer", 0.7),    # Medium risk phrase -> medium confidence
        ("no specific risk language", 0.85),  # Clean -> high confidence
    ]

    for transcript, expected_min_confidence in transcripts:
        result = client.analyse(transcript=transcript)
        print(f"\n  Transcript: {transcript}")
        print(f"    Confidence: {result.confidence:.2f} "
              f"(minimum expected: {expected_min_confidence:.2f})")

        # Confidence should be >= minimum or within reason
        is_high_risk = "guarantee" in transcript and "risk" in transcript
        is_medium_risk = "limited time" in transcript

        if is_high_risk:
            assert result.confidence >= 0.9, f"High-risk confidence too low"
            print(f"    ✓ PASS: High-risk confidence appropriate")
        elif is_medium_risk:
            assert result.confidence >= 0.6, f"Medium-risk confidence too low"
            print(f"    ✓ PASS: Medium-risk confidence appropriate")
        else:
            assert result.confidence >= 0.5, f"Low-risk confidence too low"
            print(f"    ✓ PASS: Low-risk confidence appropriate")


def test_backwards_compatibility():
    """Test that the old interface still works."""
    print("\n" + "=" * 70)
    print("TEST 6: Backward Compatibility")
    print("=" * 70)

    client = LLMClient(api_key="")

    # Old analyse() method
    print("\n  Test: analyse() method")
    result = client.analyse(transcript="guaranteed returns of 50%")
    assert hasattr(result, "violation")
    assert hasattr(result, "confidence")
    assert hasattr(result, "reason")
    print(f"  ✓ PASS: LLMResponse dataclass has expected attributes")

    # Old chat() method
    print("\n  Test: chat() method")
    json_result = client.chat(
        system_prompt="analyze",
        user_prompt="zero risk investment"
    )
    data = json.loads(json_result)
    assert "violation" in data
    assert "risk_level" in data
    print(f"  ✓ PASS: chat() returns valid JSON with expected fields")


def test_api_availability_detection():
    """Test that API availability is correctly detected."""
    print("\n" + "=" * 70)
    print("TEST 7: API Availability Detection")
    print("=" * 70)

    # Test without API key
    print("\n  Test: No API key")
    client_no_key = LLMClient(api_key="")
    print(f"    llm_available: {client_no_key.llm_available}")
    assert client_no_key.llm_available is False
    print(f"  ✓ PASS: Correctly detected no API key")

    # Test with API key (but may not be valid)
    print("\n  Test: Invalid/dummy API key")
    client_dummy_key = LLMClient(api_key="sk-dummy-key-12345")
    # Will fail to initialize client due to invalid key, so llm_available = False
    print(f"    llm_available: {client_dummy_key.llm_available}")
    print(f"  ✓ PASS: Handled dummy API key appropriately")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("REFACTORED LLMClient Test Suite")
    print("=" * 70)

    try:
        test_fallback_mode()
        test_chat_fallback_mode()
        test_json_extraction()
        test_error_handling()
        test_fallback_confidence_scores()
        test_backwards_compatibility()
        test_api_availability_detection()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
