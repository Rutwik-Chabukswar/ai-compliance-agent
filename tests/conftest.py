# tests/conftest.py
# Ensure the project root is on sys.path so `compliance_engine` is importable
# regardless of how pytest is invoked (without pip install).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
