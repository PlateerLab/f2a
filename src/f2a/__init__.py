"""f2a — File to Analysis.

A library that automatically performs descriptive statistical analysis
and visualization from various data sources.

Usage:
    >>> import f2a
    >>> report = f2a.analyze("data.csv")
    >>> report.show()
"""

from f2a._version import __version__
from f2a.core.analyzer import analyze
from f2a.core.config import AnalysisConfig

__all__ = ["__version__", "analyze", "AnalysisConfig"]
