# Evaluation Module - System Performance Analysis

## Overview

The evaluation module provides a comprehensive framework for measuring the compliance detection agent's performance across a range of labeled test cases.

**Key Features:**
- Built-in 13-case reference test suite
- Support for custom datasets (JSON format)
- Full confusion matrix metrics (TP, TN, FP, FN)
- Comprehensive performance metrics (Accuracy, Precision, Recall, F1)
- Per-example tracking with error analysis
- Multiple output formats (human-readable, JSON)
- CLI interface with flexible options

## Architecture

```
evaluate.py
├─ TestCase (dataclass)
│  └─ Holds: id, domain, transcript, expected_violation, category, result, error
│
├─ EvaluationReport (dataclass)
│  └─ Computes: accuracy, precision, recall, F1, FPR, FNR
│
├─ Functions
│  ├─ run_evaluation() → runs agent on test cases
│  ├─ load_dataset_from_json() → loads custom datasets
│  └─ print_report() → outputs results
│
└─ CLI
   └─ main() → entry point with options
```

## Metrics Explanation

### Confusion Matrix
| | Predicted Violation | Predicted Safe |
|---|---|---|
| **Expected Violation** | TP (True Positive) | FN (False Negative) |
| **Expected Safe** | FP (False Positive) | TN (True Negative) |

### Performance Metrics

**Accuracy**
- Formula: $(TP + TN) / (TP + TN + FP + FN)$
- Meaning: Overall correctness across all predictions
- Range: [0.0, 1.0] (higher is better)

**Precision**
- Formula: $TP / (TP + FP)$
- Meaning: Of predicted violations, how many were actually violations?
- Use Case: When you want to avoid false alarms
- Range: [0.0, 1.0]

**Recall (Sensitivity)**
- Formula: $TP / (TP + FN)$
- Meaning: Of actual violations, how many did we catch?
- Use Case: When you want to catch all violations
- Range: [0.0, 1.0]

**F1 Score**
- Formula: $2 * (Precision * Recall) / (Precision + Recall)$
- Meaning: Harmonic mean of precision and recall
- Use Case: Balanced metric when both false positives and negatives matter
- Range: [0.0, 1.0]

**False Positive Rate**
- Formula: $FP / (FP + TN)$
- Meaning: Of safe statements, what % were incorrectly flagged?
- Use Case: Measure overly aggressive flagging
- Range: [0.0, 1.0] (lower is better)

**False Negative Rate**
- Formula: $FN / (TP + FN)$
- Meaning: Of violations, what % were missed?
- Use Case: Measure missed detections
- Range: [0.0, 1.0] (lower is better)

## Usage

### Quick Start - Built-In Test Suite

Run the evaluation with the default 13-case test suite:
```bash
python evaluate.py
```

**Output:**
```
─────────────────────────────────────────────────────────────────────────────
  AI Compliance Detection Engine — Evaluation Report
─────────────────────────────────────────────────────────────────────────────

[PASS  ] V01  expected=violation  predicted=violation  conf=0.98
         Misleading Guarantee – explicit return guarantee

[FAIL  ] V02  expected=violation  predicted=clean      conf=0.62
         Missing Mandatory Disclosure – performance without risk disclaimer
  → False Negative: missed real violation

...

─────────────────────────────────────────────────────────────────────────────
  Summary
─────────────────────────────────────────────────────────────────────────────

  Total cases                4
  Evaluated                  4
  Errors (API/parse)         0

  ─── Confusion Matrix ───
  True Positives (TP)        8
  True Negatives (TN)        4
  False Positives (FP)       1
  False Negatives (FN)       0

  ─── Performance Metrics ─
  Accuracy                   0.9231  (92.31%)
  Precision                  0.8889  (88.89%)
  Recall                     1.0000  (100.00%)
  F1 Score                   0.9412

  ─── Error Rates ──────────
  False Positive Rate        0.2000  (20.00%)
  False Negative Rate        0.0000  (0.00%)
─────────────────────────────────────────────────────────────────────────────
```

### Custom Dataset

Load a custom JSON dataset:

```bash
python evaluate.py --dataset custom_dataset.json
```

**Dataset Format** (JSON):
```json
[
  {
    "id": "T01",
    "domain": "fintech",
    "transcript": "We guarantee 100% returns with zero risk.",
    "expected_violation": true,
    "category": "Test category"
  },
  {
    "id": "T02",
    "domain": "fintech",
    "transcript": "Our fund has historically returned 8% annually. Past performance is not guaranteed.",
    "expected_violation": false,
    "category": "Compliant example"
  }
]
```

**Required fields:**
- `transcript` (string): The text to analyze
- `expected_violation` (boolean): Ground truth label

**Optional fields:**
- `id` (string): Test case identifier (auto-generated if omitted)
- `domain` (string): Domain context (default: "fintech")
- `category` (string): Human-readable category (default: "Custom test case")

### Save Report to JSON

Generate a machine-readable JSON report:

```bash
python evaluate.py --out results.json
```

**Output Format:**
```json
{
  "summary": {
    "total": 13,
    "evaluated": 13,
    "errors": 0,
    "correct": 12,
    "accuracy": 0.9231,
    "precision": 0.8889,
    "recall": 1.0,
    "f1_score": 0.9412,
    "true_positives": 8,
    "true_negatives": 4,
    "false_positives": 1,
    "false_negatives": 0,
    "false_positive_rate": 0.2,
    "false_negative_rate": 0.0
  },
  "cases": [
    {
      "id": "V01",
      "domain": "fintech",
      "category": "Misleading Guarantee – explicit return guarantee",
      "expected_violation": true,
      "predicted_violation": true,
      "correct": true,
      "confidence": 0.98,
      "result": {...},
      "error": null
    },
    ...
  ]
}
```

### Enable Debug Output

Include debug metadata from the agent:

```bash
python evaluate.py --debug -o results.json
```

Adds debug information to each test case showing:
- Matched rules
- Reasoning summaries
- Retrieved policies (if RAG enabled)

### RAG Options

**Run WITH RAG** (default if policies available):
```bash
python evaluate.py
```

**Run WITHOUT RAG**:
```bash
python evaluate.py --no-rag
```

### Verbose Logging

Show detailed logs during evaluation:

```bash
python evaluate.py -v
```

Shows:
- INFO-level logs from compliance_engine
- Policy loading status
- Case-by-case processing

### Override LLM Model

Use a different model (requires API key):

```bash
python evaluate.py --model gpt-4o
```

## Exit Codes

The CLI uses exit codes to indicate performance:
- **0**: Success (accuracy ≥ 80%)
- **1**: Acceptable (70% ≤ accuracy < 80%)
- **2**: Poor (accuracy < 70%)

Useful for CI/CD pipelines:
```bash
python evaluate.py
if [ $? -eq 0 ]; then
  echo "Excellent performance"
elif [ $? -eq 1 ]; then
  echo "Acceptable performance"
else
  echo "Performance below threshold"
fi
```

## API Usage (Programmatic)

### Basic Evaluation

```python
from compliance_engine import ComplianceAgent
from compliance_engine.llm_client import LLMClient
from evaluate import run_evaluation, TestCase, print_report

# Create test cases
cases = [
    TestCase(
        id="T01",
        domain="fintech",
        transcript="We guarantee returns!",
        expected_violation=True,
        category="Violation"
    ),
    TestCase(
        id="T02",
        domain="fintech",
        transcript="Normal statement.",
        expected_violation=False,
        category="Safe"
    ),
]

# Initialize agent
llm_client = LLMClient()
agent = ComplianceAgent(llm_client=llm_client)

# Run evaluation
report = run_evaluation(agent, cases)

# Print results
print_report(cases, report, out_path="results.json")
```

### Access Metrics Programmatically

```python
# After run_evaluation()
print(f"Accuracy: {report.accuracy:.4f}")
print(f"Precision: {report.precision:.4f}")
print(f"Recall: {report.recall:.4f}")
print(f"F1 Score: {report.f1_score:.4f}")
print(f"False Positives: {report.false_positives}")
print(f"False Negatives: {report.false_negatives}")
```

### Load Custom Dataset

```python
from evaluate import load_dataset_from_json

cases = load_dataset_from_json("my_dataset.json")
report = run_evaluation(agent, cases)
```

## Error Handling

### Dataset Loading Errors

```python
try:
    cases = load_dataset_from_json("nonexistent.json")
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
except ValueError as e:
    print(f"Invalid dataset format: {e}")
```

**Common Issues:**
- Missing `transcript` field → ValueError
- Missing `expected_violation` field → ValueError
- JSON is not a list → ValueError
- File doesn't exist → FileNotFoundError

### Evaluation Errors

Test cases with errors are tracked separately:

```python
report = run_evaluation(agent, cases)

print(f"Errors encountered: {report.errors}")
print(f"Successfully evaluated: {report.evaluated}")
print(f"Total cases: {report.total}")

# Each case tracks its error
for tc in cases:
    if tc.error:
        print(f"Case {tc.id} failed: {tc.error}")
```

## Interpretation Guide

### Precision-Heavy Scenarios
When minimizing false positives is critical (compliance review):
```bash
python evaluate.py | grep "Precision"
# Higher precision = fewer safe statements incorrectly flagged
```

**Good:** Precision > 0.90 (90% of flagged statements truly violate)

### Recall-Heavy Scenarios
When catching all violations is critical (automated monitoring):
```bash
python evaluate.py | grep "Recall"
# Higher recall = fewer violations missed
```

**Good:** Recall > 0.95 (catches 95% of violations)

### Balanced Scenarios
When both matter equally (general compliance):
```bash
python evaluate.py | grep "F1"
# F1 balances precision and recall
```

**Good:** F1 > 0.85

## Performance Analysis

### Analyzing False Positives

```python
# Find all false positive cases
false_positives = [tc for tc in cases if tc.is_false_positive]

for tc in false_positives:
    print(f"ID: {tc.id}")
    print(f"Transcript: {tc.transcript[:80]}...")
    print(f"Category: {tc.category}")
    print(f"Confidence: {tc.result.confidence:.2f}\n")
```

### Analyzing False Negatives

```python
# Find all false negative cases (missed violations)
false_negatives = [tc for tc in cases if tc.is_false_negative]

for tc in false_negatives:
    print(f"ID: {tc.id} - MISSED VIOLATION")
    print(f"Transcript: {tc.transcript[:80]}...")
    print(f"Category: {tc.category}")
    print(f"Confidence: {tc.result.confidence:.2f}\n")
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Evaluate Compliance Agent
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python evaluate.py --out results.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: evaluation-results
          path: results.json
```

## Test Suite

Run the evaluation test suite:

```bash
pytest tests/test_evaluate.py -v
```

Tests cover:
- TestCase properties and helpers
- EvaluationReport metric calculations
- Confusion matrix computation
- Dataset loading (valid and invalid)
- Error handling

**Example:**
```bash
$ pytest tests/test_evaluate.py -v
...
tests/test_evaluate.py::TestEvaluationReport::test_accuracy_calculation PASSED
tests/test_evaluate.py::TestEvaluationReport::test_precision_calculation PASSED
tests/test_evaluate.py::TestEvaluationReport::test_recall_calculation PASSED
tests/test_evaluate.py::TestLoadDatasetFromJson::test_load_valid_dataset PASSED
...
======================== 21 passed in 0.17s ========================
```

## Files

- **`evaluate.py`**: Main evaluation module (500+ lines)
- **`tests/test_evaluate.py`**: Comprehensive test suite (21 tests)
- **`sample_dataset.json`**: Example custom dataset file

## See Also

- [ComplianceAgent Documentation](./AGENT.md)
- [Architecture Overview](./README.md)
- [Testing Guide](./README.md#testing)
