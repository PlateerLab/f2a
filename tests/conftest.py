"""pytest fixtures for f2a tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    """수치형 컬럼 위주의 샘플 DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 15000, n).round(2),
            "score": np.random.uniform(0, 100, n).round(2),
            "height": np.random.normal(170, 10, n).round(1),
        }
    )


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    """수치형 + 범주형 + 결측치가 포함된 샘플 DataFrame."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame(
        {
            "id": range(n),
            "name": [f"user_{i}" for i in range(n)],
            "age": np.random.randint(18, 80, n),
            "city": np.random.choice(["서울", "부산", "대구", "인천", "광주"], n),
            "salary": np.random.normal(50000, 15000, n).round(2),
            "rating": np.random.uniform(1, 5, n).round(1),
        }
    )
    # 결측치 삽입
    mask = np.random.random(n) < 0.1
    df.loc[mask, "salary"] = np.nan
    df.loc[np.random.random(n) < 0.05, "city"] = np.nan
    return df


@pytest.fixture
def sample_csv_path(tmp_path: Path, sample_mixed_df: pd.DataFrame) -> Path:
    """임시 CSV 파일 경로."""
    csv_path = tmp_path / "test_data.csv"
    sample_mixed_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_path(tmp_path: Path, sample_mixed_df: pd.DataFrame) -> Path:
    """임시 JSON 파일 경로."""
    json_path = tmp_path / "test_data.json"
    sample_mixed_df.to_json(json_path, orient="records", force_ascii=False)
    return json_path
