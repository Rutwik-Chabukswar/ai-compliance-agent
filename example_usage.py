#!/usr/bin/env python
"""
example_usage.py
----------------
Runnable demo of the compliance detection engine.

Requires a valid OPENAI_API_KEY in your environment.
"""

import json
import os

from compliance_engine import ComplianceEngine

DEMO_CASES = [
    {
        "label": "🚨 Clear Violation — Guaranteed Returns",
        "domain": "fintech",
        "transcript": (
            "Join our platform today and enjoy guaranteed 15% annual returns "
            "on every dollar you invest. You literally cannot lose money with us."
        ),
    },
    {
        "label": "✅ Compliant — Balanced Disclaimer",
        "domain": "fintech",
        "transcript": (
            "Historically, our diversified equity fund has averaged 8-10% "
            "annual returns over the past decade. Past performance is not a "
            "guarantee of future results, and all investments carry risk, "
            "including potential loss of principal."
        ),
    },
    {
        "label": "🤔 Ambiguous — Should NOT be flagged",
        "domain": "insurance",
        "transcript": (
            "Our premium plan gives you the best coverage options available. "
            "Most of our customers feel very well protected after switching to us."
        ),
    },
    {
        "label": "⚠️  Missing Disclosure — Crypto Performance",
        "domain": "fintech",
        "transcript": (
            "Buy into our new crypto fund today — it's up 200% in just six months. "
            "Don't miss out on the biggest opportunity of the decade!"
        ),
    },
    {
        "label": "✅ Healthcare — Qualified Supplement Claim",
        "domain": "healthcare",
        "transcript": (
            "Our supplement may support immune health as part of a balanced diet "
            "and healthy lifestyle."
        ),
    },
]


def run_demo() -> None:
    engine = ComplianceEngine()

    print("=" * 70)
    print("  AI Compliance Detection Engine — Demo")
    print("=" * 70)

    for case in DEMO_CASES:
        print(f"\n{case['label']}")
        print(f"Domain   : {case['domain']}")
        print(f"Transcript: {case['transcript'][:100]}...")
        print("-" * 50)

        result = engine.analyse(transcript=case["transcript"], domain=case["domain"])
        print(result.to_json())
        print("=" * 70)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Export it before running this demo.")
    else:
        run_demo()
