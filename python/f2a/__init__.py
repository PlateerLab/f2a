"""
f2a -- File to Analysis (Rust-powered)
=========================================

High-performance data analysis library with Rust computation core.

Usage::

    import f2a

    report = f2a.analyze("data.csv")
    report.show()           # console summary
    report.to_html("./out") # self-contained HTML report
"""

from f2a._version import __version__
from f2a.api import AnalysisConfig, analyze

__all__ = ["__version__", "analyze", "AnalysisConfig"]
