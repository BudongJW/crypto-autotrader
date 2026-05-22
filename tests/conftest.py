"""Pytest config — make ``user_data/strategies`` importable without freqtrade."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "strategies"))
