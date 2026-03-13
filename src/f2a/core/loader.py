"""데이터 로딩 모듈 — 다양한 소스에서 DataFrame을 로딩합니다.

지원 포맷:
    - **구분자 텍스트**: CSV, TSV, TXT(자동감지), DAT, TAB, FWF(고정폭)
    - **JSON 계열**: JSON, JSONL, NDJSON
    - **스프레드시트**: XLSX, XLS, XLSM, XLSB, ODS
    - **바이너리/컬럼나**: Parquet, Feather, Arrow IPC, ORC, HDF5, Pickle
    - **통계 패키지**: SAS(.sas7bdat, .xpt), Stata(.dta), SPSS(.sav, .zsav, .por)
    - **데이터베이스**: SQLite, DuckDB
    - **마크업**: XML, HTML(테이블)
    - **원격**: HTTP/HTTPS URL (확장자 기반 자동 라우팅)
    - **플랫폼**: HuggingFace Datasets (hf://...)
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd

from f2a.utils.exceptions import DataLoadError, EmptyDataError, UnsupportedFormatError
from f2a.utils.logging import get_logger
from f2a.utils.validators import HF_PREFIXES, URL_PREFIXES, detect_source_type

logger = get_logger(__name__)


class DataLoader:
    """다양한 데이터 소스에서 ``pd.DataFrame`` 을 로딩합니다.

    입력 문자열(파일 경로, URL, HuggingFace 주소 등)만으로
    포맷을 자동 감지하고 적절한 로더를 선택합니다.

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load("data.csv")
        >>> df = loader.load("hf://imdb", split="train")
        >>> df = loader.load("https://example.com/data.parquet")
        >>> df = loader.load("results.db", table="experiments")
    """

    # ── 소스 타입 → 로더 메서드 매핑 ────────────────────
    # 새 포맷 추가 시 여기에 등록하면 자동 연결됩니다.
    _LOADER_REGISTRY: dict[str, str] = {
        # 구분자 텍스트
        "csv": "_load_csv",
        "tsv": "_load_tsv",
        "delimited": "_load_delimited",
        "fwf": "_load_fwf",
        # JSON 계열
        "json": "_load_json",
        "jsonl": "_load_jsonl",
        # 스프레드시트
        "excel": "_load_excel",
        "ods": "_load_ods",
        # 바이너리 / 컬럼나
        "parquet": "_load_parquet",
        "feather": "_load_feather",
        "arrow_ipc": "_load_arrow_ipc",
        "orc": "_load_orc",
        "hdf5": "_load_hdf5",
        "pickle": "_load_pickle",
        # 통계 패키지
        "sas": "_load_sas",
        "sas_xport": "_load_sas_xport",
        "stata": "_load_stata",
        "spss": "_load_spss",
        # 데이터베이스
        "sqlite": "_load_sqlite",
        "duckdb": "_load_duckdb",
        # 마크업
        "xml": "_load_xml",
        "html": "_load_html",
        # 원격/URL
        "url_auto": "_load_url_auto",
        # HuggingFace
        "hf": "_load_huggingface",
    }

    def load(self, source: str, **kwargs: Any) -> pd.DataFrame:
        """소스 문자열을 분석하여 적절한 로더를 호출합니다.

        Args:
            source: 파일 경로, URL 또는 HuggingFace 데이터셋 주소.
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

        method_name = self._LOADER_REGISTRY.get(source_type)
        if method_name is None:
            raise UnsupportedFormatError(source, detected=source_type)

        loader_fn = getattr(self, method_name, None)
        if loader_fn is None:
            raise UnsupportedFormatError(source, detected=source_type)

        try:
            df = loader_fn(source, **kwargs)
        except (UnsupportedFormatError, DataLoadError, EmptyDataError):
            raise
        except Exception as exc:
            raise DataLoadError(source, reason=str(exc)) from exc

        if df is None or df.empty:
            raise EmptyDataError(source)

        logger.info("로딩 완료: %d행 × %d열 (%s)", len(df), len(df.columns), source_type)
        return df

    @classmethod
    def supported_formats(cls) -> list[str]:
        """지원하는 소스 타입 목록을 반환합니다."""
        return sorted(cls._LOADER_REGISTRY.keys())

    # ================================================================
    #  구분자 텍스트 (CSV / TSV / 자동감지)
    # ================================================================

    @staticmethod
    def _load_csv(source: str, **kwargs: Any) -> pd.DataFrame:
        """CSV 파일을 로딩합니다."""
        kwargs.setdefault("encoding", "utf-8")
        try:
            return pd.read_csv(source, **kwargs)
        except UnicodeDecodeError:
            kwargs["encoding"] = "cp949"  # 한국어 CSV 대응
            return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_tsv(source: str, **kwargs: Any) -> pd.DataFrame:
        """TSV 파일을 로딩합니다."""
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_delimited(source: str, **kwargs: Any) -> pd.DataFrame:
        """구분자를 자동 감지하여 텍스트 파일을 로딩합니다.

        ``csv.Sniffer`` 로 구분자를 추론하고, 실패 시 일반적인 구분자를 순차 시도합니다.
        """
        if "sep" in kwargs or "delimiter" in kwargs:
            return pd.read_csv(source, **kwargs)

        # 1단계: csv.Sniffer로 구분자 자동 감지
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(8192)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;: ")
            kwargs["sep"] = dialect.delimiter
            logger.info("구분자 자동 감지: %r", dialect.delimiter)
            return pd.read_csv(source, **kwargs)
        except csv.Error:
            pass

        # 2단계: 흔한 구분자 순차 시도
        for sep in [",", "\t", ";", "|", " "]:
            try:
                df = pd.read_csv(source, sep=sep, nrows=5, **kwargs)
                if len(df.columns) > 1:
                    logger.info("구분자 후보 확정: %r", sep)
                    return pd.read_csv(source, sep=sep, **kwargs)
            except Exception:
                continue

        # 최후: 단일 컬럼으로 로딩
        return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_fwf(source: str, **kwargs: Any) -> pd.DataFrame:
        """고정 폭(Fixed-Width Format) 파일을 로딩합니다."""
        return pd.read_fwf(source, **kwargs)

    # ================================================================
    #  JSON 계열
    # ================================================================

    @staticmethod
    def _load_json(source: str, **kwargs: Any) -> pd.DataFrame:
        """JSON 파일을 로딩합니다 (배열 또는 레코드)."""
        try:
            return pd.read_json(source, **kwargs)
        except ValueError:
            # 네스트된 JSON일 경우 normalize 시도
            import json

            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.json_normalize(data)
            elif isinstance(data, dict):
                # 키 중 하나가 데이터 배열인 경우 탐색
                for key, val in data.items():
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        logger.info("JSON 네스트 키 감지: %s", key)
                        return pd.json_normalize(val)
                return pd.json_normalize(data)
            raise

    @staticmethod
    def _load_jsonl(source: str, **kwargs: Any) -> pd.DataFrame:
        """JSONL / NDJSON 파일을 로딩합니다."""
        kwargs.setdefault("lines", True)
        return pd.read_json(source, **kwargs)

    # ================================================================
    #  스프레드시트
    # ================================================================

    @staticmethod
    def _load_excel(source: str, **kwargs: Any) -> pd.DataFrame:
        """Excel 파일을 로딩합니다 (.xlsx, .xls, .xlsm, .xlsb).

        여러 시트가 있을 경우, ``sheet_name`` 을 지정하지 않으면
        첫 번째 시트를 로딩하고 경고를 출력합니다.
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Excel 지원을 위해 'openpyxl'를 설치하세요: pip install f2a[excel]",
            ) from exc

        # xlsb 파일은 전용 엔진 필요
        if Path(source).suffix.lower() == ".xlsb":
            try:
                import pyxlsb  # noqa: F401
                kwargs.setdefault("engine", "pyxlsb")
            except ImportError as exc:
                raise DataLoadError(
                    source,
                    reason="xlsb 지원을 위해 'pyxlsb'를 설치하세요: pip install pyxlsb",
                ) from exc

        result = pd.read_excel(source, **kwargs)

        # read_excel이 dict를 반환하면 여러 시트 (sheet_name=None)
        if isinstance(result, dict):
            sheet_names = list(result.keys())
            logger.warning(
                "%d개 시트 발견: %s — 첫 번째 시트 '%s'를 사용합니다.",
                len(sheet_names),
                sheet_names,
                sheet_names[0],
            )
            return result[sheet_names[0]]
        return result

    @staticmethod
    def _load_ods(source: str, **kwargs: Any) -> pd.DataFrame:
        """ODS(OpenDocument Spreadsheet) 파일을 로딩합니다."""
        try:
            import odf  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="ODS 지원을 위해 'odfpy'를 설치하세요: pip install odfpy",
            ) from exc
        kwargs.setdefault("engine", "odf")
        return pd.read_excel(source, **kwargs)

    # ================================================================
    #  바이너리 / 컬럼나 포맷
    # ================================================================

    @staticmethod
    def _load_parquet(source: str, **kwargs: Any) -> pd.DataFrame:
        """Parquet 파일을 로딩합니다."""
        try:
            return pd.read_parquet(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Parquet 지원을 위해 'pyarrow'를 설치하세요: pip install f2a[parquet]",
            ) from exc

    @staticmethod
    def _load_feather(source: str, **kwargs: Any) -> pd.DataFrame:
        """Feather(Arrow IPC v2) 파일을 로딩합니다."""
        try:
            return pd.read_feather(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Feather 지원을 위해 'pyarrow'를 설치하세요: pip install f2a[parquet]",
            ) from exc

    @staticmethod
    def _load_arrow_ipc(source: str, **kwargs: Any) -> pd.DataFrame:
        """Apache Arrow IPC 파일을 로딩합니다."""
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Arrow IPC 지원을 위해 'pyarrow'를 설치하세요: pip install f2a[parquet]",
            ) from exc

        with open(source, "rb") as f:
            reader = ipc.open_file(f)
            table = reader.read_all()
        return table.to_pandas(**kwargs)

    @staticmethod
    def _load_orc(source: str, **kwargs: Any) -> pd.DataFrame:
        """ORC 파일을 로딩합니다."""
        try:
            return pd.read_orc(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="ORC 지원을 위해 'pyarrow'를 설치하세요: pip install f2a[parquet]",
            ) from exc

    @staticmethod
    def _load_hdf5(source: str, **kwargs: Any) -> pd.DataFrame:
        """HDF5 파일을 로딩합니다."""
        try:
            import tables  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="HDF5 지원을 위해 'tables'를 설치하세요: pip install tables",
            ) from exc

        key = kwargs.pop("key", None)
        if key:
            return pd.read_hdf(source, key=key, **kwargs)

        # key가 없으면 첫 번째 키 사용
        with pd.HDFStore(source, mode="r") as store:
            keys = store.keys()
            if not keys:
                raise DataLoadError(source, reason="HDF5 파일에 데이터셋이 없습니다.")
            if len(keys) > 1:
                logger.warning(
                    "HDF5에 %d개 키 발견: %s — 첫 번째 '%s'를 사용합니다.",
                    len(keys),
                    keys,
                    keys[0],
                )
            return pd.read_hdf(source, key=keys[0], **kwargs)

    @staticmethod
    def _load_pickle(source: str, **kwargs: Any) -> pd.DataFrame:
        """Pickle 파일을 로딩합니다.

        Warning:
            pickle은 신뢰할 수 있는 소스에서만 사용하세요.
        """
        logger.warning("pickle 로딩: 신뢰할 수 있는 소스인지 확인하세요 — %s", source)
        return pd.read_pickle(source, **kwargs)

    # ================================================================
    #  통계 패키지 포맷
    # ================================================================

    @staticmethod
    def _load_sas(source: str, **kwargs: Any) -> pd.DataFrame:
        """SAS 데이터 파일(.sas7bdat)을 로딩합니다."""
        kwargs.setdefault("format", "sas7bdat")
        return pd.read_sas(source, **kwargs)

    @staticmethod
    def _load_sas_xport(source: str, **kwargs: Any) -> pd.DataFrame:
        """SAS Transport 파일(.xpt)을 로딩합니다."""
        kwargs.setdefault("format", "xport")
        return pd.read_sas(source, **kwargs)

    @staticmethod
    def _load_stata(source: str, **kwargs: Any) -> pd.DataFrame:
        """Stata 파일(.dta)을 로딩합니다."""
        return pd.read_stata(source, **kwargs)

    @staticmethod
    def _load_spss(source: str, **kwargs: Any) -> pd.DataFrame:
        """SPSS 파일(.sav, .zsav, .por)을 로딩합니다."""
        try:
            import pyreadstat  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="SPSS 지원을 위해 'pyreadstat'를 설치하세요: pip install pyreadstat",
            ) from exc
        return pd.read_spss(source, **kwargs)

    # ================================================================
    #  데이터베이스
    # ================================================================

    @staticmethod
    def _load_sqlite(source: str, **kwargs: Any) -> pd.DataFrame:
        """SQLite 데이터베이스에서 테이블을 로딩합니다.

        Args:
            source: .db / .sqlite 파일 경로.
            **kwargs:
                table (str): 로딩할 테이블명. 미지정 시 첫 번째 테이블.
                query (str): 직접 SQL 쿼리. ``table`` 보다 우선.
        """
        import sqlite3

        table = kwargs.pop("table", None)
        query = kwargs.pop("query", None)
        conn = sqlite3.connect(source)

        try:
            if query:
                return pd.read_sql_query(query, conn, **kwargs)

            # 테이블 목록 조회
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()

            if not tables:
                raise DataLoadError(source, reason="SQLite DB에 테이블이 없습니다.")

            if table is None:
                table = tables[0]
                if len(tables) > 1:
                    logger.warning(
                        "SQLite에 %d개 테이블 발견: %s — '%s'를 사용합니다.",
                        len(tables),
                        tables,
                        table,
                    )

            if table not in tables:
                raise DataLoadError(
                    source,
                    reason=f"테이블 '{table}'이 없습니다. 사용 가능: {tables}",
                )

            return pd.read_sql_query(f'SELECT * FROM "{table}"', conn, **kwargs)
        finally:
            conn.close()

    @staticmethod
    def _load_duckdb(source: str, **kwargs: Any) -> pd.DataFrame:
        """DuckDB 데이터베이스에서 테이블을 로딩합니다."""
        try:
            import duckdb
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="DuckDB 지원을 위해 'duckdb'를 설치하세요: pip install duckdb",
            ) from exc

        table = kwargs.pop("table", None)
        query = kwargs.pop("query", None)
        conn = duckdb.connect(source, read_only=True)

        try:
            if query:
                return conn.execute(query).fetchdf()

            tables = conn.execute("SHOW TABLES").fetchdf()
            table_names = tables.iloc[:, 0].tolist() if not tables.empty else []

            if not table_names:
                raise DataLoadError(source, reason="DuckDB에 테이블이 없습니다.")

            if table is None:
                table = table_names[0]
                if len(table_names) > 1:
                    logger.warning(
                        "DuckDB에 %d개 테이블: %s — '%s'를 사용합니다.",
                        len(table_names),
                        table_names,
                        table,
                    )

            return conn.execute(f'SELECT * FROM "{table}"').fetchdf()
        finally:
            conn.close()

    # ================================================================
    #  마크업 (XML / HTML)
    # ================================================================

    @staticmethod
    def _load_xml(source: str, **kwargs: Any) -> pd.DataFrame:
        """XML 파일을 로딩합니다."""
        try:
            import lxml  # noqa: F401
        except ImportError:
            logger.info("lxml 미설치 — 기본 XML 파서를 사용합니다.")
            kwargs.setdefault("parser", "etree")
        return pd.read_xml(source, **kwargs)

    @staticmethod
    def _load_html(source: str, **kwargs: Any) -> pd.DataFrame:
        """HTML 파일에서 테이블을 추출합니다.

        여러 테이블이 있으면 가장 큰 테이블을 반환합니다.
        """
        try:
            import lxml  # noqa: F401
        except ImportError:
            logger.info("lxml 미설치 — bs4(html.parser)를 사용합니다.")
            kwargs.setdefault("flavor", "bs4")

        table_index = kwargs.pop("table_index", None)
        tables = pd.read_html(source, **kwargs)

        if not tables:
            raise DataLoadError(source, reason="HTML에서 테이블을 찾을 수 없습니다.")

        if table_index is not None:
            if table_index >= len(tables):
                raise DataLoadError(
                    source,
                    reason=f"table_index={table_index} 범위 초과 (총 {len(tables)}개 테이블)",
                )
            return tables[table_index]

        # 가장 큰 테이블 선택
        if len(tables) > 1:
            sizes = [(i, len(t) * len(t.columns)) for i, t in enumerate(tables)]
            best_idx = max(sizes, key=lambda x: x[1])[0]
            logger.warning(
                "HTML에서 %d개 테이블 발견 — 가장 큰 테이블 #%d를 사용합니다.",
                len(tables),
                best_idx,
            )
            return tables[best_idx]

        return tables[0]

    # ================================================================
    #  URL (원격 파일)
    # ================================================================

    def _load_url_auto(self, source: str, **kwargs: Any) -> pd.DataFrame:
        """URL에서 파일을 다운로드하여 로딩합니다.

        Content-Type 헤더와 URL 경로를 분석하여 포맷을 추론합니다.
        """
        import tempfile
        from urllib.parse import urlparse
        from urllib.request import urlopen, Request

        logger.info("URL 다운로드 시작: %s", source)

        req = Request(source, headers={"User-Agent": "f2a/0.1"})
        with urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            data = resp.read()

        # Content-Type 기반 포맷 추론
        ct_map = {
            "text/csv": "csv",
            "text/tab-separated-values": "tsv",
            "application/json": "json",
            "application/x-ndjson": "jsonl",
            "application/vnd.apache.parquet": "parquet",
            "application/vnd.openxmlformats": "excel",
            "application/vnd.ms-excel": "excel",
            "text/xml": "xml",
            "application/xml": "xml",
            "text/html": "html",
        }

        detected_type: str | None = None
        for ct_key, fmt in ct_map.items():
            if ct_key in content_type:
                detected_type = fmt
                break

        if detected_type is None:
            # URL 경로 확장자 확인
            from f2a.utils.validators import SUPPORTED_EXTENSIONS

            path_ext = Path(urlparse(source).path).suffix.lower()
            detected_type = SUPPORTED_EXTENSIONS.get(path_ext, "csv")

        # 임시 파일에 저장하고 해당 로더로 다시 로딩
        suffix_map = {
            "csv": ".csv",
            "tsv": ".tsv",
            "json": ".json",
            "jsonl": ".jsonl",
            "parquet": ".parquet",
            "excel": ".xlsx",
            "xml": ".xml",
            "html": ".html",
        }
        suffix = suffix_map.get(detected_type, ".tmp")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        logger.info("URL 데이터 → 임시 파일 (%s): %s", detected_type, tmp_path)

        method_name = self._LOADER_REGISTRY.get(detected_type)
        if method_name and hasattr(self, method_name):
            try:
                return getattr(self, method_name)(tmp_path, **kwargs)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # 기본: CSV로 시도
        try:
            return self._load_csv(tmp_path, **kwargs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ================================================================
    #  HuggingFace Datasets
    # ================================================================

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

        # 프리픽스 제거
        dataset_name = source
        for prefix in HF_PREFIXES:
            if dataset_name.startswith(prefix):
                dataset_name = dataset_name[len(prefix) :]
                break

        split = kwargs.pop("split", "train")
        config = kwargs.pop("config", None)

        try:
            if config:
                ds = load_dataset(dataset_name, config, split=split, **kwargs)
            else:
                ds = load_dataset(dataset_name, split=split, **kwargs)
            return ds.to_pandas()
        except Exception as exc:
            raise DataLoadError(source, reason=str(exc)) from exc
