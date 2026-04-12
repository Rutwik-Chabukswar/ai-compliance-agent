"""
tests/test_evaluate.py
----------------------
Unit tests for the evaluation module.

Tests cover:
  - Dataset loading from JSON
  - Evaluation metrics (accuracy, precision, recall, F1)
  - Confusion matrix computation
  - Error analysis
  - Report generation
"""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

from evaluate import (
    TestCase,
    EvaluationReport,
    run_evaluation,
    load_dataset_from_json,
)
from compliance_engine import ComplianceAgent, ComplianceResult


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def sample_test_cases() -> list:
    """Create sample test cases for evaluation."""
    return [
        TestCase(
            id="T01",
            domain="fintech",
            transcript="We guarantee 100% returns!",
            expected_violation=True,
            category="High-risk violation",
        ),
        TestCase(
            id="T02",
            domain="fintech",
            transcript="This is a simple statement.",
            expected_violation=False,
            category="Compliant",
        ),
        TestCase(
            id="T03",
            domain="insurance",
            transcript="We offer comprehensive coverage with no exclusions.",
            expected_violation=True,
            category="Coverage violation",
        ),
        TestCase(
            id="T04",
            domain="fintech",
            transcript="Past performance is not a guarantee.",
            expected_violation=False,
            category="Compliant with disclaimer",
        ),
    ]


@pytest.fixture()
def sample_dataset_file() -> str:
    """Create a temporary sample dataset JSON file."""
    data = [
        {
            "id": "test1",
            "domain": "fintech",
            "transcript": "Guaranteed returns!",
            "expected_violation": True,
            "category": "Violation",
        },
        {
            "id": "test2",
            "domain": "fintech",
            "transcript": "Normal statement",
            "expected_violation": False,
            "category": "Safe",
        },
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        return f.name


@pytest.fixture()
def mock_agent() -> MagicMock:
    """Create a mock ComplianceAgent."""
    agent = MagicMock(spec=ComplianceAgent)
    
    def mock_analyse(transcript: str, domain: str):
        # Simple mock: flag if "guarantee" or "exclusion" in transcript
        violation = "guarantee" in transcript.lower() or "exclusion" in transcript.lower()
        return ComplianceResult(
            violation=violation,
            risk_level="high" if violation else "low",
            confidence=0.95 if violation else 0.90,
            reason="Test reason",
            suggestion="Test suggestion",
        )
    
    agent.analyse.side_effect = mock_analyse
    return agent


# =========================================================================
# TestCase Tests
# =========================================================================


class TestTestCaseProperties:
    """Test TestCase dataclass properties."""

    def test_test_case_creation(self):
        """TestCase should initialize with all fields."""
        tc = TestCase(
            id="T01",
            domain="fintech",
            transcript="Test",
            expected_violation=True,
            category="Test category",
        )
        assert tc.id == "T01"
        assert tc.domain == "fintech"
        assert tc.expected_violation is True
        assert tc.result is None
        assert tc.error is None

    def test_predicted_violation_property(self):
        """predicted_violation should return result.violation."""
        tc = TestCase(
            id="T01",
            domain="fintech",
            transcript="Test",
            expected_violation=True,
            category="Test",
        )
        
        result = ComplianceResult(
            violation=True,
            risk_level="high",
            confidence=0.9,
            reason="Test",
            suggestion="Test",
        )
        tc.result = result
        
        assert tc.predicted_violation is True

    def test_correct_property_true(self):
        """correct should be True when prediction matches expected."""
        tc = TestCase(
            id="T01",
            domain="fintech",
            transcript="Test",
            expected_violation=True,
            category="Test",
        )
        
        tc.result = ComplianceResult(
            violation=True,
            risk_level="high",
            confidence=0.9,
            reason="Test",
            suggestion="Test",
        )
        
        assert tc.correct is True

    def test_correct_property_false(self):
        """correct should be False when prediction doesn't match."""
        tc = TestCase(
            id="T01",
            domain="fintech",
            transcript="Test",
            expected_violation=True,
            category="Test",
        )
        
        tc.result = ComplianceResult(
            violation=False,
            risk_level="low",
            confidence=0.9,
            reason="Test",
            suggestion="Test",
        )
        
        assert tc.correct is False

    def test_is_false_positive(self):
        """is_false_positive should be True for predicted but not expected."""
        tc = TestCase(
            id="T01",
            domain="fintech",
            transcript="Test",
            expected_violation=False,
            category="Test",
        )
        
        tc.result = ComplianceResult(
            violation=True,  # Predicted violation
            risk_level="high",
            confidence=0.9,
            reason="Test",
            suggestion="Test",
        )
        
        assert tc.is_false_positive is True

    def test_is_false_negative(self):
        """is_false_negative should be True for expected but not predicted."""
        tc = TestCase(
            id="T01",
            domain="fintech",
            transcript="Test",
            expected_violation=True,
            category="Test",
        )
        
        tc.result = ComplianceResult(
            violation=False,  # Missed violation
            risk_level="low",
            confidence=0.9,
            reason="Test",
            suggestion="Test",
        )
        
        assert tc.is_false_negative is True


# =========================================================================
# EvaluationReport Tests
# =========================================================================


class TestEvaluationReport:
    """Test EvaluationReport metrics computation."""

    def test_perfect_evaluation(self):
        """All metrics should be 1.0 for perfect predictions."""
        report = EvaluationReport(
            total=4,
            evaluated=4,
            errors=0,
            correct=4,
            true_positives=2,
            true_negatives=2,
            false_positives=0,
            false_negatives=0,
        )
        
        assert report.accuracy == 1.0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0

    def test_accuracy_calculation(self):
        """Accuracy should be (TP + TN) / Total."""
        report = EvaluationReport(
            total=4,
            evaluated=4,
            errors=0,
            correct=3,
            true_positives=2,
            true_negatives=1,
            false_positives=1,
            false_negatives=0,
        )
        
        assert report.accuracy == 0.75  # 3/4

    def test_precision_calculation(self):
        """Precision should be TP / (TP + FP)."""
        report = EvaluationReport(
            total=4,
            evaluated=4,
            errors=0,
            correct=2,
            true_positives=2,
            true_negatives=0,
            false_positives=1,
            false_negatives=1,
        )
        
        # Precision = 2 / (2 + 1) = 0.666...
        assert pytest.approx(report.precision, 0.001) == 2/3

    def test_recall_calculation(self):
        """Recall should be TP / (TP + FN)."""
        report = EvaluationReport(
            total=4,
            evaluated=4,
            errors=0,
            correct=2,
            true_positives=2,
            true_negatives=0,
            false_positives=0,
            false_negatives=1,
        )
        
        # Recall = 2 / (2 + 1) = 0.666...
        assert pytest.approx(report.recall, 0.001) == 2/3

    def test_f1_score_calculation(self):
        """F1 should be 2 * (P * R) / (P + R)."""
        report = EvaluationReport(
            total=4,
            evaluated=4,
            errors=0,
            correct=2,
            true_positives=2,
            true_negatives=0,
            false_positives=1,
            false_negatives=1,
        )
        
        # Precision = 2/3, Recall = 2/3
        # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
        assert pytest.approx(report.f1_score, 0.001) == 2/3

    def test_false_positive_rate(self):
        """FPR should be FP / (FP + TN)."""
        report = EvaluationReport(
            total=4,
            evaluated=4,
            errors=0,
            correct=2,
            true_positives=2,
            true_negatives=1,
            false_positives=1,
            false_negatives=0,
        )
        
        # FPR = 1 / (1 + 1) = 0.5
        assert report.false_positive_rate == 0.5

    def test_false_negative_rate(self):
        """FNR should be FN / (TP + FN)."""
        report = EvaluationReport(
            total=4,
            evaluated=4,
            errors=0,
            correct=2,
            true_positives=2,
            true_negatives=1,
            false_positives=0,
            false_negatives=1,
        )
        
        # FNR = 1 / (2 + 1) = 0.333...
        assert pytest.approx(report.false_negative_rate, 0.001) == 1/3


# =========================================================================
# Run Evaluation Tests
# =========================================================================


class TestRunEvaluation:
    """Test the run_evaluation function."""

    def test_run_evaluation_perfect(
        self, mock_agent: MagicMock, sample_test_cases: list
    ):
        """run_evaluation should compute correct metrics."""
        # Set up mock to return correct predictions
        def perfect_analyse(transcript: str, domain: str):
            # Predict based on "guarantee" keyword
            violation = "guarantee" in transcript.lower()
            return ComplianceResult(
                violation=violation,
                risk_level="high" if violation else "low",
                confidence=0.95,
                reason="Test",
                suggestion="Test",
            )

        mock_agent.analyse.side_effect = perfect_analyse

        report = run_evaluation(mock_agent, sample_test_cases)

        # T01: "guarantee 100%..." → predicted=True, expected=True → TP
        # T02: "simple statement" → predicted=False, expected=False → TN
        # T03: "no exclusions" (no guarantee) → predicted=False, expected=True → FN
        # T04: "not a guarantee" → predicted=True, expected=False → FP
        assert report.total == 4
        assert report.evaluated == 4
        assert report.errors == 0
        assert report.true_positives == 1   # T01
        assert report.true_negatives == 1   # T02
        assert report.false_positives == 1  # T04
        assert report.false_negatives == 1  # T03

    def test_run_evaluation_with_errors(self, sample_test_cases: list):
        """run_evaluation should handle errors gracefully."""
        agent = MagicMock(spec=ComplianceAgent)
        agent.analyse.side_effect = Exception("Test error")

        report = run_evaluation(agent, sample_test_cases)

        assert report.total == 4
        assert report.evaluated == 0
        assert report.errors == 4


# =========================================================================
# Dataset Loading Tests
# =========================================================================


class TestLoadDatasetFromJson:
    """Test the load_dataset_from_json function."""

    def test_load_valid_dataset(self, sample_dataset_file: str):
        """Should load valid JSON dataset successfully."""
        cases = load_dataset_from_json(sample_dataset_file)
        
        assert len(cases) == 2
        assert cases[0].id == "test1"
        assert cases[0].domain == "fintech"
        assert cases[0].expected_violation is True
        assert cases[1].id == "test2"
        assert cases[1].expected_violation is False

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_dataset_from_json("/nonexistent/file.json")

    def test_load_invalid_json_not_list(self):
        """Should raise ValueError if JSON is not a list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="must be a list"):
                    load_dataset_from_json(f.name)
            finally:
                os.unlink(f.name)

    def test_load_missing_transcript_field(self):
        """Should raise ValueError for missing transcript field."""
        data = [{"id": "T01", "domain": "fintech", "expected_violation": True}]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="missing required field"):
                    load_dataset_from_json(f.name)
            finally:
                os.unlink(f.name)

    def test_load_missing_expected_violation_field(self):
        """Should raise ValueError for missing expected_violation field."""
        data = [{"id": "T01", "domain": "fintech", "transcript": "Test"}]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="missing required field"):
                    load_dataset_from_json(f.name)
            finally:
                os.unlink(f.name)

    def test_load_with_defaults(self):
        """Should populate default values for optional fields."""
        data = [
            {
                "transcript": "Test",
                "expected_violation": True,
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            
            try:
                cases = load_dataset_from_json(f.name)
                assert len(cases) == 1
                assert cases[0].id == "T01"
                assert cases[0].domain == "fintech"
                assert cases[0].category == "Custom test case"
            finally:
                os.unlink(f.name)
