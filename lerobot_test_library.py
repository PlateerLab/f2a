"""Quick test script: generate f2a report for lerobot/roboturk."""

import warnings

warnings.filterwarnings("ignore")

import f2a

result = f2a.analyze("lerobot/roboturk")
path = result.to_html("output")
print(f"Report saved: {path}")
