#!/usr/bin/env python
"""
cli.py
------
Command-line interface for the compliance detection engine.

Usage examples
--------------
# From a string:
python cli.py --transcript "We guarantee 15% returns" --domain fintech

# From a file:
python cli.py --file transcript.txt --domain insurance

# Pretty-print JSON output (default):
python cli.py --transcript "..." --domain fintech --pretty

# Machine-readable compact JSON:
python cli.py --transcript "..." --domain fintech --compact
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from compliance_engine import ComplianceEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compliance-check",
        description="AI-powered compliance analysis for regulated industry transcripts.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--transcript", "-t",
        type=str,
        help="Transcript text passed directly on the command line.",
    )
    source.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to a plain-text file containing the transcript.",
    )
    source.add_argument(
        "--audio", "-a",
        type=Path,
        help="Path to an audio file (e.g., .wav) to transcribe before analysis.",
    )
    parser.add_argument(
        "--domain", "-d",
        required=True,
        type=str,
        help="Regulatory domain (e.g. fintech, insurance, healthcare).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default LLM model (e.g. gpt-4o, gpt-4o-mini).",
    )
    parser.add_argument(
        "--audio-provider",
        type=str,
        choices=["whisper", "sarvam"],
        default=None,
        help="Audio transcription provider (default: whisper).",
    )
    parser.add_argument(
        "--rag-context",
        type=str,
        default=None,
        help="Optional policy text to prepend as RAG context.",
    )
    output_fmt = parser.add_mutually_exclusive_group()
    output_fmt.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default).",
    )
    output_fmt.add_argument(
        "--compact",
        action="store_true",
        default=False,
        help="Output compact single-line JSON.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Simulate streaming by processing the transcript chunk-by-chunk.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve transcript text
    if args.file:
        if not args.file.exists():
            parser.error(f"File not found: {args.file}")
        transcript = args.file.read_text(encoding="utf-8")
    elif args.audio:
        if not args.audio.exists():
            parser.error(f"Audio file not found: {args.audio}")
        from compliance_engine.audio.transcriber import transcribe_audio
        
        print(f"[•] Transcribing audio file: {args.audio} ...", file=sys.stderr)
        transcript_data = transcribe_audio(args.audio, provider=args.audio_provider)
        transcript = transcript_data["text"]
        
        # Debug Output
        print("\n--- Audio Transcript ---", file=sys.stderr)
        if transcript_data.get("confidence") is not None:
            print(f"Confidence: {transcript_data['confidence']:.2f}", file=sys.stderr)
        print(transcript, file=sys.stderr)
        print("------------------------\n", file=sys.stderr)
    else:
        transcript = args.transcript

    # Build engine (optionally override model)
    engine_kwargs: dict = {}
    if args.model:
        from compliance_engine.llm_client import LLMClient
        engine_kwargs["llm_client"] = LLMClient(model=args.model)

    engine = ComplianceEngine(**engine_kwargs)

    if args.stream:
        from compliance_engine.streaming import StreamingProcessor
        from compliance_engine.streaming.segmenter import simulate_segmentation
        
        processor = StreamingProcessor(engine, args.domain)
        # Deeply analyze punctuation, pauses, and speakers simulating Celerio VAD
        segments = simulate_segmentation(transcript)
        
        print("\n=== Streaming Analysis Initialized ===", file=sys.stderr)
        final_result = None
        for i, chunk_data in enumerate(segments):
            text = chunk_data.get("text", "")
            speaker = chunk_data.get("speaker", "unknown").title()
            print(f"\n[Chunk {i+1}] [{speaker}]: '{text}'", file=sys.stderr)
            
            result = processor.process_chunk(chunk_data)
            if result:
                print(f"  -> Violation: {result.violation} | Risk: {result.risk_level} | Conf: {result.confidence:.2f}", file=sys.stderr)
                if result.violation:
                    print(f"  -> Reason: {result.reason}", file=sys.stderr)
                final_result = result
                
        print("\n=== Stream Complete ===\n", file=sys.stderr)
        
        indent = None if args.compact else 2
        output = json.dumps(final_result.to_dict() if final_result else {}, indent=indent)
        print(output)
        sys.exit(1 if (final_result and final_result.violation) else 0)
    else:
        # Standard synchronous execution
        result = engine.analyse(
            transcript=transcript,
            domain=args.domain,
            rag_context=args.rag_context,
        )

        indent = None if args.compact else 2
        output = json.dumps(result.to_dict(), indent=indent)
        print(output)
        sys.exit(1 if result.violation else 0)


if __name__ == "__main__":
    main()
