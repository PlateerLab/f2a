"""DataLoader 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from f2a.core.loader import DataLoader
from f2a.utils.exceptions import EmptyDataError, UnsupportedFormatError
from f2a.utils.validators import detect_source_type


class TestDetectSourceType:
    """소스 타입 감지 테스트."""

    def test_csv(self) -> None:
        assert detect_source_type("data.csv") == "csv"

    def test_tsv(self) -> None:
        assert detect_source_type("data.tsv") == "tsv"

    def test_json(self) -> None:
        assert detect_source_type("data.json") == "json"

    def test_jsonl(self) -> None:
        assert detect_source_type("data.jsonl") == "jsonl"

    def test_parquet(self) -> None:
        assert detect_source_type("data.parquet") == "parquet"

    def test_excel_xlsx(self) -> None:
        assert detect_source_type("data.xlsx") == "excel"

    def test_hf_prefix(self) -> None:
        assert detect_source_type("hf://imdb") == "hf"

    def test_unsupported(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            detect_source_type("data.xyz")


class TestDataLoader:
    """DataLoader 테스트."""

    def test_load_csv(self, sample_csv_path: Path) -> None:
        loader = DataLoader()
        df = loader.load(str(sample_csv_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_json(self, sample_json_path: Path) -> None:
        loader = DataLoader()
        df = loader.load(str(sample_json_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_nonexistent_file(self) -> None:
        loader = DataLoader()
        with pytest.raises(Exception):
            loader.load("nonexistent_file.csv")
