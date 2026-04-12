"""
demo_streaming_simulator.py
---------------------------
Demonstrates the new StreamingComplianceProcessor simulation layer.
"""

from compliance_engine.agent import ComplianceAgent
from compliance_engine.llm_client import LLMClient
from compliance_engine.streaming.simulator import StreamingComplianceProcessor

def run_demo():
    print("=" * 70)
    print("  SIMULATED STREAMING COMPLIANCE DETECTION DEMO")
    print("=" * 70)

    # Initialize components
    # The client automatically falls back to FREE_MODE if no API key is set
    llm_client = LLMClient()
    agent = ComplianceAgent(llm_client=llm_client)

    # Initialize our new simulated streaming processor
    # chunk_size=5 simulates receiving small segments sequentially
    processor = StreamingComplianceProcessor(agent=agent, chunk_size=5)

    # Transcript begins smoothly and hits a red flag at the end
    transcript = (
        "Welcome to our investment advisory. We offer a variety of services to manage your portfolio. "
        "Historically our funds have done well. Now we absolutely guarantee 100% returns for everyone."
    )

    print(f"Full Transcript:\n'{transcript}'\n")
    print("Starting simulated streaming analysis...\n")

    # Begin the simulated stream processing
    stream_generator = processor.process_stream(transcript, domain="fintech")
    
    for result in stream_generator:
        print(f"[Chunk] {result['chunk']}")
        print(f"        Violation: {result['violation']}")
        if result['violation']:
            print(f"        Suggestion: {result['suggestion']}")
            print("        🚨 EARLY ALERT TRIGGERED! Halting stream... 🚨\n")
    
    print("End of Stream.")

if __name__ == "__main__":
    run_demo()
