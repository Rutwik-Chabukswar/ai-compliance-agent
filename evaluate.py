"""
evaluate.py
-----------
Offline evaluation harness for the AI Compliance Detection Engine.

Defines 13 labelled test cases spanning clear violations, safe compliant
statements, and ambiguous edge cases. Runs the engine on all cases and
prints a structured accuracy report.

Usage
-----
    # Standard run (uses OPENAI_API_KEY from env)
    python evaluate.py

    # Enable debug metadata on each result
    python evaluate.py --debug

    # Override model
    python evaluate.py --model gpt-4o

    # Save report to a JSON file
    python evaluate.py --out results.json
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from compliance_engine import ComplianceEngine, ComplianceResult

# Silence INFO-level logs during evaluation so the report is readable.
logging.getLogger("compliance_engine").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single labelled evaluation example."""

    id: str
    domain: str
    transcript: str
    expected_violation: bool  # Ground-truth label
    category: str             # Human-readable category tag (for the report)

    # Runtime results (filled after evaluation)
    result: Optional[ComplianceResult] = field(default=None, repr=False)
    error: Optional[str] = field(default=None, repr=False)

    @property
    def predicted_violation(self) -> Optional[bool]:
        return self.result.violation if self.result else None

    @property
    def correct(self) -> Optional[bool]:
        if self.predicted_violation is None:
            return None
        return self.predicted_violation == self.expected_violation

    @property
    def is_false_positive(self) -> bool:
        """Predicted violation but should be clean."""
        return self.predicted_violation is True and self.expected_violation is False

    @property
    def is_false_negative(self) -> bool:
        """Missed a real violation."""
        return self.predicted_violation is False and self.expected_violation is True


# ---------------------------------------------------------------------------
# Ground-truth test suite (13 cases)
# ---------------------------------------------------------------------------

TEST_CASES: List[TestCase] = [
    # ---- VIOLATIONS (expected_violation = True) ----
    TestCase(
        id="V01",
        domain="fintech",
        transcript=(
            "Our platform guarantees a 15% annual return on every portfolio. "
            "You literally cannot lose money investing with us."
        ),
        expected_violation=True,
        category="Misleading Guarantee – explicit return guarantee",
    ),
    TestCase(
        id="V02",
        domain="fintech",
        transcript=(
            "Buy into our new crypto fund today — it's up 300% in just three months! "
            "Don't miss the biggest opportunity of the decade."
        ),
        expected_violation=True,
        category="Missing Mandatory Disclosure – performance without risk disclaimer",
    ),
    TestCase(
        id="V03",
        domain="insurance",
        transcript=(
            "With our policy, you are fully covered for every possible incident — "
            "no exclusions, no deductibles, no questions asked."
        ),
        expected_violation=True,
        category="Misleading Guarantee – absolute coverage promise",
    ),
    TestCase(
        id="V04",
        domain="healthcare",
        transcript=(
            "Our supplement cures type 2 diabetes within 30 days, "
            "clinically proven and FDA-approved for diabetes treatment."
        ),
        expected_violation=True,
        category="Illegal Claim – false cure and fabricated FDA approval",
    ),
    TestCase(
        id="V05",
        domain="fintech",
        transcript=(
            "We are a licensed broker-dealer. Our algorithm has beaten the S&P 500 "
            "every single year for the past decade — sign up and get the same results."
        ),
        expected_violation=True,
        category="Misleading Guarantee – implied guaranteed future performance",
    ),
    TestCase(
        id="V06",
        domain="fintech",
        transcript=(
            "This is not investment advice, but buying this token right now is "
            "essentially risk-free. Every analyst agrees: guaranteed upside."
        ),
        expected_violation=True,
        category="Misleading Guarantee – 'risk-free' claim with disclaimer contradiction",
    ),
    TestCase(
        id="V07",
        domain="healthcare",
        transcript=(
            "Our IV drip therapy reverses ageing at the cellular level "
            "and has been shown to extend lifespan by 20 years in clinical trials."
        ),
        expected_violation=True,
        category="Illegal Claim – fabricated clinical trial outcome",
    ),

    # ---- COMPLIANT / SAFE (expected_violation = False) ----
    TestCase(
        id="S01",
        domain="fintech",
        transcript=(
            "Historically, our diversified equity fund has averaged 8-10% annual "
            "returns over the past decade. Past performance is not a guarantee of "
            "future results. All investments carry risk, including potential loss of "
            "principal."
        ),
        expected_violation=False,
        category="Safe – historical data with required disclaimer",
    ),
    TestCase(
        id="S02",
        domain="insurance",
        transcript=(
            "Our premium health plan gives you access to a broad network of "
            "specialists. Coverage limits and exclusions apply — please review "
            "your policy document for full details."
        ),
        expected_violation=False,
        category="Safe – general description with coverage caveat",
    ),
    TestCase(
        id="S03",
        domain="healthcare",
        transcript=(
            "Our supplement may support immune health as part of a balanced diet "
            "and healthy lifestyle. These statements have not been evaluated by the "
            "FDA and are not intended to diagnose, treat, cure, or prevent any disease."
        ),
        expected_violation=False,
        category="Safe – qualified supplement claim with FDA disclaimer",
    ),

    # ---- AMBIGUOUS (expected_violation = False — should NOT be flagged) ----
    TestCase(
        id="A01",
        domain="insurance",
        transcript=(
            "Our premium plan gives you the best coverage options available. "
            "Most of our customers feel very well protected after switching."
        ),
        expected_violation=False,
        category="Ambiguous – superlative marketing language, no guarantee",
    ),
    TestCase(
        id="A02",
        domain="fintech",
        transcript=(
            "Our robo-advisor uses advanced AI to optimise your portfolio. "
            "Many investors have seen strong returns in recent years."
        ),
        expected_violation=False,
        category="Ambiguous – vague performance anecdote, no specific guarantee",
    ),
    TestCase(
        id="A03",
        domain="healthcare",
        transcript=(
            "Thousands of customers report feeling more energetic within two weeks "
            "of starting our wellness program. Results may vary."
        ),
        expected_violation=False,
        category="Ambiguous – anecdotal testimonial with 'results may vary' qualifier",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Aggregated metrics computed after running all test cases."""

    total: int
    evaluated: int
    errors: int
    correct: int
    false_positives: int
    false_negatives: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.evaluated if self.evaluated else 0.0

    @property
    def false_positive_rate(self) -> float:
        safe_cases = self.evaluated - (self.correct + self.false_negatives
                                       - max(0, self.correct - self.false_negatives))
        total_safe = sum(1 for tc in TEST_CASES if not tc.expected_violation)
        return self.false_positives / total_safe if total_safe else 0.0

    @property
    def false_negative_rate(self) -> float:
        total_violations = sum(1 for tc in TEST_CASES if tc.expected_violation)
        return self.false_negatives / total_violations if total_violations else 0.0


def run_evaluation(
    engine: ComplianceEngine,
    cases: List[TestCase],
    debug: bool = False,
) -> EvaluationReport:
    """
    Run the engine on every test case and return aggregated metrics.

    Parameters
    ----------
    engine:
        Initialised :class:`ComplianceEngine` instance.
    cases:
        List of labelled :class:`TestCase` objects.
    debug:
        Forward ``debug=True`` to ``engine.analyse()`` to capture debug info.

    Returns
    -------
    EvaluationReport
    """
    errors = 0

    for tc in cases:
        try:
            tc.result = engine.analyse(
                transcript=tc.transcript,
                domain=tc.domain,
                debug=debug,
            )
        except Exception as exc:  # noqa: BLE001
            tc.error = str(exc)
            errors += 1
            logging.warning("Case %s failed: %s", tc.id, exc)

    evaluated = [tc for tc in cases if tc.result is not None]
    correct = sum(1 for tc in evaluated if tc.correct)
    false_positives = sum(1 for tc in evaluated if tc.is_false_positive)
    false_negatives = sum(1 for tc in evaluated if tc.is_false_negative)

    return EvaluationReport(
        total=len(cases),
        evaluated=len(evaluated),
        errors=errors,
        correct=correct,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

_USE_COLOR = sys.stdout.isatty()


def _c(text: str, *codes: str) -> str:
    """Apply ANSI colour codes when writing to a terminal."""
    if not _USE_COLOR:
        return text
    return "".join(codes) + text + _RESET


def print_report(
    cases: List[TestCase],
    report: EvaluationReport,
    debug: bool = False,
    out_path: Optional[str] = None,
) -> None:
    """
    Print a structured evaluation report to stdout and optionally write JSON.

    Parameters
    ----------
    cases:
        Evaluated test cases (with ``.result`` populated).
    report:
        Aggregated metrics.
    debug:
        Whether debug info was requested (controls an extra column).
    out_path:
        If set, write a machine-readable JSON report to this path.
    """
    sep = "─" * 78

    print()
    print(_c(sep, _BOLD))
    print(_c("  AI Compliance Detection Engine — Evaluation Report", _BOLD, _CYAN))
    print(_c(sep, _BOLD))

    # ── Per-case results ────────────────────────────────────────────────────
    for tc in cases:
        if tc.result is None:
            status = _c("ERROR ", _RED)
            detail = f"  ⚠  {tc.error}"
        elif tc.correct:
            status = _c("PASS  ", _GREEN)
            detail = ""
        else:
            if tc.is_false_positive:
                status = _c("FAIL  ", _RED)
                detail = _c("  → False Positive: flagged safe statement", _YELLOW)
            else:
                status = _c("FAIL  ", _RED)
                detail = _c("  → False Negative: missed real violation", _YELLOW)

        predicted = (
            _c("violation", _RED) if tc.predicted_violation
            else _c("clean    ", _GREEN)
            if tc.predicted_violation is not None
            else _c("N/A      ", _DIM)
        )
        expected = (
            _c("violation", _RED) if tc.expected_violation else _c("clean    ", _GREEN)
        )

        confidence_str = ""
        if tc.result:
            conf = tc.result.confidence
            col = _GREEN if conf >= 0.7 else _YELLOW if conf >= 0.5 else _RED
            confidence_str = _c(f"  conf={conf:.2f}", col)

        print(
            f"[{status}] {tc.id}  "
            f"expected={expected}  predicted={predicted}"
            f"{confidence_str}"
        )
        print(_c(f"         {tc.category}", _DIM))
        if detail:
            print(detail)

        # Debug sub-block
        if debug and tc.result and tc.result.debug:
            dbg = tc.result.debug
            print(
                _c(
                    f"         rule={dbg.matched_rule}  "
                    f"summary=\"{dbg.reasoning_summary}\"",
                    _DIM,
                )
            )
        print()

    # ── Summary ─────────────────────────────────────────────────────────────
    print(_c(sep, _BOLD))
    print(_c("  Summary", _BOLD))
    print(_c(sep, _BOLD))

    acc_color = _GREEN if report.accuracy >= 0.8 else _YELLOW if report.accuracy >= 0.6 else _RED

    rows = [
        ("Total cases",       str(report.total)),
        ("Evaluated",         str(report.evaluated)),
        ("Errors (API/parse)", str(report.errors)),
        ("Correct",           str(report.correct)),
        ("Accuracy",          _c(f"{report.accuracy:.4f}  ({report.accuracy * 100:.1f}%)", acc_color)),
        ("False Positives",   _c(str(report.false_positives), _YELLOW if report.false_positives else _GREEN)),
        ("False Negatives",   _c(str(report.false_negatives), _YELLOW if report.false_negatives else _GREEN)),
    ]
    for label, value in rows:
        print(f"  {label:<25} {value}")

    print(_c(sep, _BOLD))
    print()

    # ── JSON export ─────────────────────────────────────────────────────────
    if out_path:
        payload = {
            "summary": {
                "total": report.total,
                "evaluated": report.evaluated,
                "errors": report.errors,
                "correct": report.correct,
                "accuracy": round(report.accuracy, 4),
                "false_positives": report.false_positives,
                "false_negatives": report.false_negatives,
            },
            "cases": [
                {
                    "id": tc.id,
                    "domain": tc.domain,
                    "category": tc.category,
                    "expected_violation": tc.expected_violation,
                    "predicted_violation": tc.predicted_violation,
                    "correct": tc.correct,
                    "result": tc.result.to_dict() if tc.result else None,
                    "error": tc.error,
                }
                for tc in cases
            ],
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  JSON report saved to: {out_path}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate",
        description="Run the compliance engine against a labelled test suite.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model (e.g. gpt-4o, gpt-4o-mini).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode on each engine call.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="FILE",
        help="Write machine-readable JSON report to FILE.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show INFO-level engine logs during evaluation.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.verbose:
        logging.getLogger("compliance_engine").setLevel(logging.INFO)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY is not set. Export it before running evaluate.py.")
        sys.exit(1)

    engine_kwargs: dict = {}
    if args.model:
        from compliance_engine.llm_client import LLMClient
        engine_kwargs["llm_client"] = LLMClient(model=args.model)

    engine = ComplianceEngine(**engine_kwargs)

    print(f"\nRunning {len(TEST_CASES)} test cases …")
    report = run_evaluation(engine, TEST_CASES, debug=args.debug)
    print_report(TEST_CASES, report, debug=args.debug, out_path=args.out)

    # Exit 0 only when accuracy is above 70%
    sys.exit(0 if report.accuracy >= 0.70 else 1)


if __name__ == "__main__":
    main()
