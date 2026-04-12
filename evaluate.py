"""
evaluate.py
-----------
Offline evaluation harness for the AI Compliance Detection System.

Runs the ComplianceAgent against labelled test cases and computes comprehensive
metrics: accuracy, precision, recall, false positives, and false negatives.

Supports:
  - Built-in 13-case test suite
  - Custom JSON dataset files
  - Multiple output formats (human-readable, JSON)
  - Error analysis and per-case tracking
  - Multiple evaluation modes

Usage
-----
    # Standard run (built-in test cases)
    python evaluate.py

    # Load custom dataset
    python evaluate.py --dataset custom_dataset.json

    # Enable debug metadata on each result
    python evaluate.py --debug

    # Save report to JSON
    python evaluate.py --out results.json

    # Verbose logging
    python evaluate.py -v
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from compliance_engine import ComplianceAgent, ComplianceResult
from compliance_engine.llm_client import LLMClient
from compliance_engine.rag import PolicyRetriever, load_policies_from_directory
from compliance_engine.config import POLICIES_DIR

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
    true_positives: int
    true_negatives: int

    @property
    def accuracy(self) -> float:
        """(TP + TN) / Total Predictions"""
        return (self.true_positives + self.true_negatives) / self.evaluated if self.evaluated else 0.0

    @property
    def precision(self) -> float:
        """TP / (TP + FP) — Of predicted violations, how many were correct?"""
        total_predicted_violations = self.true_positives + self.false_positives
        return self.true_positives / total_predicted_violations if total_predicted_violations else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN) — Of actual violations, how many did we catch?"""
        total_actual_violations = self.true_positives + self.false_negatives
        return self.true_positives / total_actual_violations if total_actual_violations else 0.0

    @property
    def f1_score(self) -> float:
        """2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def false_positive_rate(self) -> float:
        """FP / (FP + TN) — Of safe statements, how many were incorrectly flagged?"""
        total_safe = self.false_positives + self.true_negatives
        return self.false_positives / total_safe if total_safe else 0.0

    @property
    def false_negative_rate(self) -> float:
        """FN / (TP + FN) — Of violations, how many were missed?"""
        total_violations = self.true_positives + self.false_negatives
        return self.false_negatives / total_violations if total_violations else 0.0


def run_evaluation(
    agent: ComplianceAgent,
    cases: List[TestCase],
    debug: bool = False,
) -> EvaluationReport:
    """
    Run the agent on every test case and return aggregated metrics.

    Computes accuracy, precision, recall, F1 score, and error rates.

    Parameters
    ----------
    agent:
        Initialised :class:`ComplianceAgent` instance.
    cases:
        List of labelled :class:`TestCase` objects.
    debug:
        If True, includes debug metadata in results.

    Returns
    -------
    EvaluationReport
        Comprehensive metrics including TP, TN, FP, FN.
    """
    errors = 0

    for tc in cases:
        try:
            tc.result = agent.analyse(
                transcript=tc.transcript,
                domain=tc.domain,
            )
        except Exception as exc:  # noqa: BLE001
            tc.error = str(exc)
            errors += 1
            logging.warning("Case %s failed: %s", tc.id, exc)

    evaluated = [tc for tc in cases if tc.result is not None]
    
    # Compute confusion matrix components
    true_positives = sum(1 for tc in evaluated if tc.predicted_violation and tc.expected_violation)
    true_negatives = sum(1 for tc in evaluated if not tc.predicted_violation and not tc.expected_violation)
    false_positives = sum(1 for tc in evaluated if tc.predicted_violation and not tc.expected_violation)
    false_negatives = sum(1 for tc in evaluated if not tc.predicted_violation and tc.expected_violation)

    return EvaluationReport(
        total=len(cases),
        evaluated=len(evaluated),
        errors=errors,
        correct=true_positives + true_negatives,
        true_positives=true_positives,
        true_negatives=true_negatives,
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
    prec_color = _GREEN if report.precision >= 0.8 else _YELLOW if report.precision >= 0.6 else _RED
    recall_color = _GREEN if report.recall >= 0.8 else _YELLOW if report.recall >= 0.6 else _RED
    f1_color = _GREEN if report.f1_score >= 0.7 else _YELLOW if report.f1_score >= 0.5 else _RED

    rows = [
        ("Total cases",             str(report.total)),
        ("Evaluated",               str(report.evaluated)),
        ("Errors (API/parse)",      str(report.errors)),
        ("",                        ""),
        ("─── Confusion Matrix ───", ""),
        ("True Positives (TP)",     str(report.true_positives)),
        ("True Negatives (TN)",     str(report.true_negatives)),
        ("False Positives (FP)",    _c(str(report.false_positives), _YELLOW if report.false_positives else _GREEN)),
        ("False Negatives (FN)",    _c(str(report.false_negatives), _YELLOW if report.false_negatives else _GREEN)),
        ("",                        ""),
        ("─── Performance Metrics ─", ""),
        ("Accuracy",                _c(f"{report.accuracy:.4f}  ({report.accuracy * 100:.1f}%)", acc_color)),
        ("Precision",               _c(f"{report.precision:.4f}  ({report.precision * 100:.1f}%)", prec_color)),
        ("Recall",                  _c(f"{report.recall:.4f}  ({report.recall * 100:.1f}%)", recall_color)),
        ("F1 Score",                _c(f"{report.f1_score:.4f}", f1_color)),
        ("",                        ""),
        ("─── Error Rates ──────────", ""),
        ("False Positive Rate",     _c(f"{report.false_positive_rate:.4f}  ({report.false_positive_rate * 100:.1f}%)", _YELLOW if report.false_positive_rate > 0.1 else _GREEN)),
        ("False Negative Rate",     _c(f"{report.false_negative_rate:.4f}  ({report.false_negative_rate * 100:.1f}%)", _YELLOW if report.false_negative_rate > 0.1 else _GREEN)),
    ]
    for label, value in rows:
        if label == "":
            continue
        if "───" in label:
            print(_c(f"  {label}", _DIM))
        else:
            print(f"  {label:<27} {value}")

    print(_c(sep, _BOLD))
    print()

    # ── JSON export ─────────────────────────────────────────────────────────
    if out_path:
        payload = {
            "summary": {
                "total": report.total,
                "evaluated": report.evaluated,
                "errors": report.errors,
                "correct": report.true_positives + report.true_negatives,
                "accuracy": round(report.accuracy, 6),
                "precision": round(report.precision, 6),
                "recall": round(report.recall, 6),
                "f1_score": round(report.f1_score, 6),
                "true_positives": report.true_positives,
                "true_negatives": report.true_negatives,
                "false_positives": report.false_positives,
                "false_negatives": report.false_negatives,
                "false_positive_rate": round(report.false_positive_rate, 6),
                "false_negative_rate": round(report.false_negative_rate, 6),
            },
            "cases": [
                {
                    "id": tc.id,
                    "domain": tc.domain,
                    "category": tc.category,
                    "expected_violation": tc.expected_violation,
                    "predicted_violation": tc.predicted_violation,
                    "correct": tc.correct,
                    "confidence": tc.result.confidence if tc.result else None,
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
        description="Evaluate the compliance agent against labelled test cases.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to custom dataset JSON file (if not provided, uses built-in 13-case suite).",
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
        help="Enable debug mode on each agent call.",
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
        help="Show INFO-level logs during evaluation.",
    )
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (run agent without policy retrieval).",
    )
    return p


def load_dataset_from_json(path: str) -> List[TestCase]:
    """
    Load a custom evaluation dataset from a JSON file.

    Expected format:
    [
        {
            "id": "T01",
            "domain": "fintech",
            "transcript": "...",
            "expected_violation": true,
            "category": "Violation type"
        },
        ...
    ]

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    List[TestCase]
        Loaded test cases.

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    ValueError
        If JSON format is invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of test cases")

    cases = []
    for i, item in enumerate(data):
        try:
            case = TestCase(
                id=item.get("id", f"T{i+1:02d}"),
                domain=item.get("domain", "fintech"),
                transcript=item["transcript"],
                expected_violation=item["expected_violation"],
                category=item.get("category", "Custom test case"),
            )
            cases.append(case)
        except KeyError as e:
            raise ValueError(
                f"Test case {i} missing required field: {e}"
            ) from e
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Test case {i} has invalid format: {e}"
            ) from e

    logging.info("Loaded %d test cases from %s", len(cases), path)
    return cases


def main() -> None:
    args = build_parser().parse_args()

    if args.verbose:
        logging.getLogger("compliance_engine").setLevel(logging.INFO)
    else:
        logging.getLogger("compliance_engine").setLevel(logging.WARNING)

    # Load dataset
    if args.dataset:
        try:
            cases = load_dataset_from_json(args.dataset)
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Dataset loading failed: {e}")
            sys.exit(1)
    else:
        cases = TEST_CASES

    # Initialize agent components
    llm_client = LLMClient(model=args.model) if args.model else LLMClient()

    retriever = None
    if not args.no_rag:
        try:
            policies = load_policies_from_directory(POLICIES_DIR)
            if policies:
                retriever = PolicyRetriever(policies)
                logging.info("Loaded %d policies for RAG", len(policies))
        except Exception as e:
            logging.warning("Failed to load policies: %s. Running without RAG.", str(e))

    # Initialize agent
    agent = ComplianceAgent(
        llm_client=llm_client,
        retriever=retriever,
        use_fallback_on_error=True,
    )

    # Run evaluation
    print(f"\nRunning evaluation on {len(cases)} test cases…")
    if retriever:
        print(f"  ✓ RAG enabled ({len(retriever.chunks)} policies)")
    else:
        print(f"  ○ RAG disabled")
    print()

    report = run_evaluation(agent, cases, debug=args.debug)
    print_report(cases, report, debug=args.debug, out_path=args.out)

    # Exit code based on accuracy
    if report.accuracy >= 0.80:
        sys.exit(0)
    elif report.accuracy >= 0.70:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
