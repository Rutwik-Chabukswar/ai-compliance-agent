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
    else:
        transcript = args.transcript

    # Build engine (optionally override model)
    engine_kwargs: dict = {}
    if args.model:
        from compliance_engine.llm_client import LLMClient
        engine_kwargs["llm_client"] = LLMClient(model=args.model)

    engine = ComplianceEngine(**engine_kwargs)

    result = engine.analyse(
        transcript=transcript,
        domain=args.domain,
        rag_context=args.rag_context,
    )

    indent = None if args.compact else 2
    output = json.dumps(result.to_dict(), indent=indent)
    print(output)

    # Exit with non-zero code if a violation was detected (useful in CI pipelines)
    sys.exit(1 if result.violation else 0)


if __name__ == "__main__":
    main()
