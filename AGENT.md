# ComplianceAgent Orchestration Layer

## Overview

The `ComplianceAgent` is a lightweight orchestration layer that dynamically selects and executes the most appropriate compliance detection strategy based on transcript content.

Instead of always using the same approach, the agent makes intelligent decisions:

- **RULE_ONLY**: Fast detection for obvious violations
- **DIRECT_LLM**: Standard LLM analysis
- **RAG_AUGMENTED**: LLM analysis with policy context

## Architecture

```
Transcript Input
    ↓
Strategy Decision (_decide_strategy)
    ↓
┌─────────────────────────────────────┐
│  Which strategy should we use?      │
│  ├─ Rule-only keywords? → RULE_ONLY │
│  ├─ Domain keywords? → RAG_AUGMENTED│
│  └─ Default → DIRECT_LLM            │
└─────────────────────────────────────┘
    ↓
Strategy Execution
    ├─ _run_rules()          (rule-based detection)
    ├─ _run_llm()            (standard LLM)
    └─ _run_rag_pipeline()   (LLM + retrieved context)
    ↓
ComplianceResult Output
```

## Strategy Decision Logic

The agent uses keyword matching to decide the strategy:

### RULE_ONLY Keywords (Highest Priority)
Triggers fast rule-based detection for obvious compliance violations:
- `"guarantee"`, `"guaranteed"`, `"guarantees"`
- `"100%"`
- `"risk-free"`, `"riskfree"`
- `"zero risk"`, `"no risk"`

**Example**: "We guarantee 100% returns" → RULE_ONLY (immediate detection)

### RAG_AUGMENTED Keywords
Triggers policy-context-augmented analysis for domain-specific content:
- `"investment"`, `"policy"`, `"policies"`
- `"returns"`, `"insurance"`, `"premium"`, `"coverage"`
- `"compliance"`, `"regulation"`, `"sec"`, `"finra"`

**Example**: "Our investment policy covers returns" → RAG_AUGMENTED (uses retrieved policies)

### DIRECT_LLM (Default)
Used when no special keywords are detected:

**Example**: "This is a simple statement" → DIRECT_LLM (standard analysis)

## Usage

### Basic Usage (LLM Only)
```python
from compliance_engine import ComplianceAgent
from compliance_engine.llm_client import LLMClient

# Initialize
llm_client = LLMClient()
agent = ComplianceAgent(llm_client=llm_client, retriever=None)

# Analyze
result = agent.analyse(
    transcript="We offer market-linked returns.",
    domain="fintech"
)

print(result.to_json())
```

### With RAG Support
```python
from compliance_engine import ComplianceAgent
from compliance_engine.llm_client import LLMClient
from compliance_engine.rag import PolicyRetriever, load_policies_from_directory
from compliance_engine.config import POLICIES_DIR

# Initialize with RAG
llm_client = LLMClient()
policies = load_policies_from_directory(POLICIES_DIR)
retriever = PolicyRetriever(policies)

agent = ComplianceAgent(
    llm_client=llm_client,
    retriever=retriever,
    use_fallback_on_error=True  # Optional: enable error fallback
)

# Analyze
result = agent.analyse(
    transcript="Our investment policy ensures coverage.",
    domain="fintech"
)

if result.debug:
    print(f"Retrieved policies: {result.debug.get('retrieved_policies')}")
    print(f"Grounding score: {result.debug.get('grounding_score')}")
print(result.to_json())
```

## Response Structure

All strategies return a consistent `ComplianceResult`:

```json
{
  "violation": true,
  "risk_level": "high",
  "confidence": 0.92,
  "reason": "Detected guaranteed return claim, which is prohibited.",
  "suggestion": "Remove guaranteed language and add appropriate risk disclaimers.",
  "debug": {
    "matched_rule": "RULE_BASED_DETECTION",
    "reasoning_summary": "Rule-based: guaranteed return + risk keywords",
    "retrieved_policies": ["fintech_policy.txt (score=0.85)"],
    "grounding_score": 0.85
  }
}
```

**Fields**:
- `violation`: Boolean indicating if a compliance issue was detected
- `risk_level`: `"low"`, `"medium"`, or `"high"`
- `confidence`: Float [0.0, 1.0] indicating decision certainty
- `reason`: Human-readable explanation
- `suggestion`: Recommended corrective action
- `debug`: Optional metadata (only when strategies retrieve policies)

## Strategy Details

### Strategy 1: RULE_ONLY

**When Used**: Highest-confidence rule violations detected

**Flow**:
1. Extract keywords from transcript
2. Match against rule violations
3. Return immediate detection

**Speed**: ⚡ Fastest (no LLM, no RAG)

**Confidence**: Very high (95%+)

**Example**:
```python
agent.analyse("Guaranteed 100% returns!", "fintech")
# → RULE_ONLY strategy
# → High confidence (0.95)
```

### Strategy 2: DIRECT_LLM

**When Used**: No special keywords detected

**Flow**:
1. Prepare system prompt with domain context
2. Call LLM with transcript
3. Return LLM decision

**Speed**: 🚀 Standard (LLM call)

**Confidence**: Depends on LLM confidence

**Example**:
```python
agent.analyse("Please analyze this statement.", "fintech")
# → DIRECT_LLM strategy
# → Standard LLM analysis
```

### Strategy 3: RAG_AUGMENTED

**When Used**: Domain-specific or compliance-related keywords detected

**Flow**:
1. Extract keywords from transcript
2. Retrieve top-k relevant policies
3. Format policies into context
4. Call LLM with augmented prompt
5. Blend LLM confidence with grounding score
6. Return decision with debug metadata

**Speed**: 🐢 Slower (RAG retrieval + LLM)

**Confidence**: High (blended LLM + grounding)

**Grounding Score Calculation**:
```
grounding_score = avg(retrieved_chunk_scores) + retrieval_boost
retrieval_boost = min(num_chunks * 0.05, 0.2)
final_confidence = (llm_confidence + grounding_score) / 2
```

**Example**:
```python
agent.analyse("What about investment policy coverage?", "fintech")
# → RAG_AUGMENTED strategy
# → Retrieves 3 relevant policies
# → Blends confidence
# → Returns with debug info
```

## Configuration

### Initialize Agent
```python
agent = ComplianceAgent(
    llm_client=llm_client,           # Required
    retriever=retriever,              # Optional (enables RAG)
    use_fallback_on_error=True       # Optional (fallback on failures)
)
```

**Parameters**:
- `llm_client` (required): LLMClient instance
- `retriever` (optional): PolicyRetriever instance for RAG
- `use_fallback_on_error` (default=True): Enable fallback to rule-based on errors

### Analyze Transcript
```python
result = agent.analyse(
    transcript="...",                 # Non-empty string (required)
    domain="fintech"                 # Non-empty string (required)
)
```

**Parameters**:
- `transcript`: The text to analyze (raises ValueError if empty)
- `domain`: Domain context like "fintech", "insurance", "healthcare" (raises ValueError if empty)

**Returns**: `ComplianceResult` with violation status and metadata

## Error Handling

### Default Behavior (Fallback Enabled)
```python
agent = ComplianceAgent(llm_client, retriever, use_fallback_on_error=True)

try:
    result = agent.analyse(transcript, domain)
    # If LLM fails, automatically falls back to rule-based detection
except ValueError as e:
    # Input validation error (empty transcript/domain)
    print(f"Invalid input: {e}")
```

### Strict Mode (No Fallback)
```python
agent = ComplianceAgent(llm_client, retriever, use_fallback_on_error=False)

try:
    result = agent.analyse(transcript, domain)
except Exception as e:
    # LLM errors will propagate (no fallback)
    print(f"Analysis failed: {e}")
```

## Logging

The agent provides comprehensive logging at each step:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see:
# DEBUG - Strategy decision: RAG_AUGMENTED (keyword='investment')
# DEBUG - Retrieved 3 policy chunks from RAG
# DEBUG - RAG_AUGMENTED strategy completed: violation=False, confidence=0.82
```

**Log Levels**:
- `DEBUG`: Strategy selection, retrieval details, intermediate steps
- `INFO`: Strategy execution summaries, final results
- `WARNING`: Fallback activations, missing components
- `ERROR`: Analysis failures

## Performance Characteristics

| Strategy | Speed | Confidence | Use Case |
|----------|-------|-----------|----------|
| RULE_ONLY | ⚡ Fast | ⭐⭐⭐⭐⭐ Very High | Obvious violations |
| DIRECT_LLM | 🚀 Standard | ⭐⭐⭐⭐ High | Generic analysis |
| RAG_AUGMENTED | 🐢 Slower | ⭐⭐⭐⭐⭐ Very High | Domain-specific |

**Typical Latencies** (estimated):
- RULE_ONLY: ~10ms (no external calls)
- DIRECT_LLM: 500-2000ms (LLM call)
- RAG_AUGMENTED: 600-2500ms (retrieval + LLM call)

## Advanced Usage

### Custom Strategy Logic
To implement custom strategy selection, subclass ComplianceAgent:

```python
class CustomAgent(ComplianceAgent):
    def analyse(self, transcript: str, domain: str):
        # Your custom logic here
        strategy = self.custom_decide_strategy(transcript)
        return super().analyse(transcript, domain)
```

### Result Processing
```python
result = agent.analyse(transcript, domain)

# Access all fields
if result.violation:
    print(f"Risk: {result.risk_level}")
    if result.debug:
        for policy in result.debug.retrieved_policies:
            print(f"  - {policy}")

# Convert to dict/JSON
result_dict = result.to_dict(include_debug=True)
result_json = result.to_json(indent=2)
```

## Testing

### Unit Tests
Run all tests:
```bash
pytest tests/test_agent.py -v
```

Expected: 34 tests passing

### Integration Tests
Verify with real LLM:
```bash
pytest tests/test_agent.py::TestComplianceAgent -v
```

### Example
Run the example script:
```bash
python example_agent_usage.py
```

## Troubleshooting

### Strategy Not Being Selected as Expected
```python
from compliance_engine.agent import _decide_strategy

# Debug the decision
strategy = _decide_strategy("your transcript")
print(f"Selected strategy: {strategy}")
```

### Low Confidence Scores
Check the debug output:
```python
result = agent.analyse(transcript, domain)
if result.debug:
    print(f"Grounding score: {result.debug.grounding_score}")
    print(f"Retrieved policies: {len(result.debug.retrieved_policies)}")
```

### Retriever Not Being Used
Ensure retriever is initialized and has policies:
```python
retriever = PolicyRetriever(policies)
print(f"Loaded {len(retriever.chunks)} policies")

agent = ComplianceAgent(llm_client, retriever)
```

## Files

- **`compliance_engine/agent.py`**: Main agent orchestration logic
- **`tests/test_agent.py`**: Comprehensive test suite (34 tests)
- **`example_agent_usage.py`**: Example usage patterns
- **`compliance_engine/__init__.py`**: Public API exports

## Integration with ComplianceEngine

The agent layer sits above the existing ComplianceEngine but doesn't replace it:

```
ComplianceAgent (NEW - orchestration layer)
    ↓
ComplianceEngine (existing - core logic)
    ├─ LLMClient (existing)
    └─ PolicyRetriever (existing)
```

You can use either:
- **ComplianceEngine** directly for more control
- **ComplianceAgent** for automatic strategy selection

## Next Steps

1. ✅ Build lightweight orchestration layer
2. ✅ Implement dynamic strategy selection
3. ✅ Add comprehensive logging
4. ✅ Create full test suite (96 total tests passing)
5. **Possible Future**: Multi-round conversations, confidence thresholds, custom rules
