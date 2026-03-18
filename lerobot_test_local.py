"""Quick test script: generate f2a report for lerobot/roboturk (local version)."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure the local f2a package is used instead of the installed one
sys.path.insert(0, str(Path(__file__).resolve().parent))

import f2a  # noqa: E402

result = f2a.analyze("lerobot/roboturk")
path = result.to_html("output")
print(f"Report saved: {path}")
