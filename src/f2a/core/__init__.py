"""Core module — 데이터 로딩, 분석 오케스트레이션, 스키마 추론."""

from f2a.core.loader import DataLoader
from f2a.core.analyzer import analyze, Analyzer
from f2a.core.schema import DataSchema, infer_schema

__all__ = ["DataLoader", "analyze", "Analyzer", "DataSchema", "infer_schema"]
