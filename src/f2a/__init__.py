"""f2a — File to Analysis.

다양한 데이터 소스를 입력받아 기술 통계 분석 및 시각화를 자동으로 수행하는 라이브러리입니다.

Usage:
    >>> import f2a
    >>> report = f2a.analyze("data.csv")
    >>> report.show()
"""

from f2a._version import __version__
from f2a.core.analyzer import analyze

__all__ = ["__version__", "analyze"]
