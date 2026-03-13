"""데이터 로딩 모듈 — 다양한 소스에서 DataFrame을 로딩합니다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from f2a.utils.exceptions import DataLoadError, EmptyDataError, UnsupportedFormatError
from f2a.utils.logging import get_logger
from f2a.utils.validators import HF_PREFIXES, detect_source_type

logger = get_logger(__name__)


class DataLoader:
    """다양한 데이터 소스에서 ``pd.DataFrame`` 을 로딩합니다.

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load("data.csv")
        >>> df = loader.load("hf://imdb", split="train")
    """

    def load(self, source: str, **kwargs: Any) -> pd.DataFrame:
        """소스 문자열을 분석하여 적절한 로더를 호출합니다.

        Args:
            source: 파일 경로 또는 HuggingFace 데이터셋 주소.
            **kwargs: 각 로더에 전달할 추가 인자.

        Returns:
            로딩된 DataFrame.

        Raises:
            UnsupportedFormatError: 지원하지 않는 포맷.
            DataLoadError: 로딩 중 오류 발생.
            EmptyDataError: 로딩 결과가 빈 DataFrame.
        """
        source_type = detect_source_type(source)
        logger.info("소스 타입 감지: %s → %s", source, source_type)

        loader_map = {
            "csv": self._load_csv,
            "tsv": self._load_tsv,
            "json": self._load_json,
            "jsonl": self._load_jsonl,
            "parquet": self._load_parquet,
            "excel": self._load_excel,
            "hf": self._load_huggingface,
        }

        loader_fn = loader_map.get(source_type)
        if loader_fn is None:
            raise UnsupportedFormatError(source, detected=source_type)

        try:
            df = loader_fn(source, **kwargs)
        except (UnsupportedFormatError, DataLoadError, EmptyDataError):
            raise
        except Exception as exc:
            raise DataLoadError(source, reason=str(exc)) from exc

        if df.empty:
            raise EmptyDataError(source)

        logger.info("로딩 완료: %d행 × %d열", len(df), len(df.columns))
        return df

    # ── 개별 로더 ────────────────────────────────────────

    @staticmethod
    def _load_csv(source: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_tsv(source: str, **kwargs: Any) -> pd.DataFrame:
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_json(source: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_json(source, **kwargs)

    @staticmethod
    def _load_jsonl(source: str, **kwargs: Any) -> pd.DataFrame:
        kwargs.setdefault("lines", True)
        return pd.read_json(source, **kwargs)

    @staticmethod
    def _load_parquet(source: str, **kwargs: Any) -> pd.DataFrame:
        try:
            return pd.read_parquet(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Parquet 지원을 위해 'pyarrow'를 설치하세요: pip install f2a[parquet]",
            ) from exc

    @staticmethod
    def _load_excel(source: str, **kwargs: Any) -> pd.DataFrame:
        try:
            return pd.read_excel(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Excel 지원을 위해 'openpyxl'를 설치하세요: pip install f2a[excel]",
            ) from exc

    @staticmethod
    def _load_huggingface(source: str, **kwargs: Any) -> pd.DataFrame:
        """HuggingFace 데이터셋을 로딩합니다."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="HuggingFace 지원을 위해 'datasets'를 설치하세요: pip install f2a[hf]",
            ) from exc

        # hf:// 프리픽스 제거
        dataset_name = source
        for prefix in HF_PREFIXES:
            if dataset_name.startswith(prefix):
                dataset_name = dataset_name[len(prefix) :]
                break

        split = kwargs.pop("split", "train")
        try:
            ds = load_dataset(dataset_name, split=split, **kwargs)
            return ds.to_pandas()
        except Exception as exc:
            raise DataLoadError(source, reason=str(exc)) from exc
