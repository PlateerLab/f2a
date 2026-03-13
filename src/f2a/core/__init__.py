"""Core module — data loading, analysis orchestration, and schema inference."""

from f2a.core.analyzer import Analyzer, analyze
from f2a.core.config import AnalysisConfig
from f2a.core.loader import DataLoader
from f2a.core.preprocessor import Preprocessor, PreprocessingResult
from f2a.core.schema import DataSchema, infer_schema

__all__ = [
    "AnalysisConfig",
    "Analyzer",
    "DataLoader",
    "DataSchema",
    "Preprocessor",
    "PreprocessingResult",
    "analyze",
    "infer_schema",
]
