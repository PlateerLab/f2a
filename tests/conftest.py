"""pytest fixtures for f2a tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def mixed_df() -> pd.DataFrame:
    """Session-scoped mixed-type DataFrame with missing values."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame(
        {
            "id": range(n),
            "age": np.random.randint(18, 80, n),
            "income": np.random.lognormal(10, 1, n),
            "score": np.random.normal(75, 15, n),
            "category": np.random.choice(["A", "B", "C", "D"], n),
            "city": np.random.choice(["Seoul", "Busan", "Daegu", "Incheon"], n),
            "passed": np.random.choice([True, False], n),
        }
    )
    df.loc[np.random.choice(n, 20, replace=False), "income"] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), "score"] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), "city"] = np.nan
    return df


@pytest.fixture(scope="session")
def numeric_df() -> pd.DataFrame:
    """Session-scoped numeric-only DataFrame."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame(
        {
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.normal(5, 2, n),
            "x3": np.random.exponential(2, n),
            "x4": np.random.uniform(-10, 10, n),
            "x5": np.random.poisson(3, n).astype(float),
        }
    )


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp directory for test data files."""
    return tmp_path_factory.mktemp("f2a_test_data")


@pytest.fixture(scope="session")
def csv_path(data_dir: Path, mixed_df: pd.DataFrame) -> Path:
    p = data_dir / "mixed_data.csv"
    mixed_df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def tsv_path(data_dir: Path, mixed_df: pd.DataFrame) -> Path:
    p = data_dir / "mixed_data.tsv"
    mixed_df.to_csv(p, index=False, sep="\t")
    return p


@pytest.fixture(scope="session")
def json_path(data_dir: Path, mixed_df: pd.DataFrame) -> Path:
    p = data_dir / "mixed_data.json"
    mixed_df.to_json(p, orient="records", force_ascii=False)
    return p


@pytest.fixture(scope="session")
def jsonl_path(data_dir: Path, mixed_df: pd.DataFrame) -> Path:
    p = data_dir / "mixed_data.jsonl"
    mixed_df.to_json(p, orient="records", lines=True, force_ascii=False)
    return p


@pytest.fixture(scope="session")
def parquet_path(data_dir: Path, mixed_df: pd.DataFrame) -> Path:
    p = data_dir / "mixed_data.parquet"
    mixed_df.to_parquet(p, index=False)
    return p


@pytest.fixture(scope="session")
def numeric_csv_path(data_dir: Path, numeric_df: pd.DataFrame) -> Path:
    p = data_dir / "numeric_only.csv"
    numeric_df.to_csv(p, index=False)
    return p
