"""Data loading module — loads DataFrames from various sources.

Supported formats:
    - **Delimited text**: CSV, TSV, TXT (auto-detect), DAT, TAB, FWF (fixed-width)
    - **JSON family**: JSON, JSONL, NDJSON
    - **Spreadsheets**: XLSX, XLS, XLSM, XLSB, ODS
    - **Binary/columnar**: Parquet, Feather, Arrow IPC, ORC, HDF5, Pickle
    - **Statistical packages**: SAS (.sas7bdat, .xpt), Stata (.dta), SPSS (.sav, .zsav, .por)
    - **Databases**: SQLite, DuckDB
    - **Markup**: XML, HTML (tables)
    - **Remote**: HTTP/HTTPS URL (auto-routing by extension)
    - **Platforms**: HuggingFace Datasets (hf://...)
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd

from f2a.utils.exceptions import DataLoadError, EmptyDataError, UnsupportedFormatError
from f2a.utils.logging import get_logger
from f2a.utils.validators import HF_PREFIXES, HF_URL_PATTERN, URL_PREFIXES, detect_source_type

logger = get_logger(__name__)


class DataLoader:
    """Load ``pd.DataFrame`` from various data sources.

    Automatically detects the format from the input string (file path, URL,
    HuggingFace address, etc.) and selects the appropriate loader.

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load("data.csv")
        >>> df = loader.load("hf://imdb", split="train")
        >>> df = loader.load("https://example.com/data.parquet")
        >>> df = loader.load("results.db", table="experiments")
    """

    # ── Source type → loader method mapping ────────────────────
    # Register new formats here; they will be auto-routed.
    _LOADER_REGISTRY: dict[str, str] = {
        # Delimited text
        "csv": "_load_csv",
        "tsv": "_load_tsv",
        "delimited": "_load_delimited",
        "fwf": "_load_fwf",
        # JSON family
        "json": "_load_json",
        "jsonl": "_load_jsonl",
        # Spreadsheets
        "excel": "_load_excel",
        "ods": "_load_ods",
        # Binary / columnar
        "parquet": "_load_parquet",
        "feather": "_load_feather",
        "arrow_ipc": "_load_arrow_ipc",
        "orc": "_load_orc",
        "hdf5": "_load_hdf5",
        "pickle": "_load_pickle",
        # Statistical packages
        "sas": "_load_sas",
        "sas_xport": "_load_sas_xport",
        "stata": "_load_stata",
        "spss": "_load_spss",
        # Databases
        "sqlite": "_load_sqlite",
        "duckdb": "_load_duckdb",
        # Markup
        "xml": "_load_xml",
        "html": "_load_html",
        # Remote/URL
        "url_auto": "_load_url_auto",
        # HuggingFace
        "hf": "_load_huggingface",
    }

    def load(self, source: str, **kwargs: Any) -> pd.DataFrame:
        """Analyze the source string and call the appropriate loader.

        Args:
            source: File path, URL, or HuggingFace dataset address.
            **kwargs: Additional arguments passed to the loader.

        Returns:
            Loaded DataFrame.

        Raises:
            UnsupportedFormatError: Unsupported format.
            DataLoadError: Error during loading.
            EmptyDataError: Loaded result is an empty DataFrame.
        """
        source_type = detect_source_type(source)
        logger.info("Source type detected: %s → %s", source, source_type)

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

        logger.info("Loading complete: %d rows × %d cols (%s)", len(df), len(df.columns), source_type)
        return df

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Return list of supported source types."""
        return sorted(cls._LOADER_REGISTRY.keys())

    # ================================================================
    #  Delimited text (CSV / TSV / auto-detect)
    # ================================================================

    @staticmethod
    def _load_csv(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file."""
        kwargs.setdefault("encoding", "utf-8")
        try:
            return pd.read_csv(source, **kwargs)
        except UnicodeDecodeError:
            kwargs["encoding"] = "cp949"  # fallback for Korean CSV
            return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_tsv(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a TSV file."""
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_delimited(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a text file with auto-detected delimiter.

        Uses ``csv.Sniffer`` to infer the delimiter; falls back to common delimiters.
        """
        if "sep" in kwargs or "delimiter" in kwargs:
            return pd.read_csv(source, **kwargs)

        # Step 1: auto-detect delimiter with csv.Sniffer
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(8192)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;: ")
            kwargs["sep"] = dialect.delimiter
            logger.info("Delimiter auto-detected: %r", dialect.delimiter)
            return pd.read_csv(source, **kwargs)
        except csv.Error:
            pass

        # Step 2: try common delimiters sequentially
        for sep in [",", "\t", ";", "|", " "]:
            try:
                df = pd.read_csv(source, sep=sep, nrows=5, **kwargs)
                if len(df.columns) > 1:
                    logger.info("Delimiter confirmed: %r", sep)
                    return pd.read_csv(source, sep=sep, **kwargs)
            except Exception:
                continue

        # Last resort: load as single column
        return pd.read_csv(source, **kwargs)

    @staticmethod
    def _load_fwf(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a fixed-width format (FWF) file."""
        return pd.read_fwf(source, **kwargs)

    # ================================================================
    #  JSON family
    # ================================================================

    @staticmethod
    def _load_json(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a JSON file (array or records)."""
        try:
            return pd.read_json(source, **kwargs)
        except ValueError:
            # Attempt normalize for nested JSON
            import json

            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.json_normalize(data)
            elif isinstance(data, dict):
                # Look for a key containing an array of records
                for key, val in data.items():
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        logger.info("Nested JSON key detected: %s", key)
                        return pd.json_normalize(val)
                return pd.json_normalize(data)
            raise

    @staticmethod
    def _load_jsonl(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a JSONL / NDJSON file."""
        kwargs.setdefault("lines", True)
        return pd.read_json(source, **kwargs)

    # ================================================================
    #  Spreadsheets
    # ================================================================

    @staticmethod
    def _load_excel(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load an Excel file (.xlsx, .xls, .xlsm, .xlsb).

        If multiple sheets exist and ``sheet_name`` is not specified,
        the first sheet is loaded with a warning.
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'openpyxl' for Excel support: pip install f2a[excel]",
            ) from exc

        # xlsb files require a dedicated engine
        if Path(source).suffix.lower() == ".xlsb":
            try:
                import pyxlsb  # noqa: F401
                kwargs.setdefault("engine", "pyxlsb")
            except ImportError as exc:
                raise DataLoadError(
                    source,
                    reason="Install 'pyxlsb' for xlsb support: pip install pyxlsb",
                ) from exc

        result = pd.read_excel(source, **kwargs)

        # read_excel returns a dict when multiple sheets exist (sheet_name=None)
        if isinstance(result, dict):
            sheet_names = list(result.keys())
            logger.warning(
                "%d sheets found: %s — using first sheet '%s'.",
                len(sheet_names),
                sheet_names,
                sheet_names[0],
            )
            return result[sheet_names[0]]
        return result

    @staticmethod
    def _load_ods(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load an ODS (OpenDocument Spreadsheet) file."""
        try:
            import odf  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'odfpy' for ODS support: pip install odfpy",
            ) from exc
        kwargs.setdefault("engine", "odf")
        return pd.read_excel(source, **kwargs)

    # ================================================================
    #  Binary / columnar formats
    # ================================================================

    @staticmethod
    def _load_parquet(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a Parquet file."""
        try:
            return pd.read_parquet(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'pyarrow' for Parquet support: pip install f2a[parquet]",
            ) from exc

    @staticmethod
    def _load_feather(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a Feather (Arrow IPC v2) file."""
        try:
            return pd.read_feather(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'pyarrow' for Feather support: pip install f2a[parquet]",
            ) from exc

    @staticmethod
    def _load_arrow_ipc(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load an Apache Arrow IPC file."""
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'pyarrow' for Arrow IPC support: pip install f2a[parquet]",
            ) from exc

        with open(source, "rb") as f:
            reader = ipc.open_file(f)
            table = reader.read_all()
        return table.to_pandas(**kwargs)

    @staticmethod
    def _load_orc(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load an ORC file."""
        try:
            return pd.read_orc(source, **kwargs)
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'pyarrow' for ORC support: pip install f2a[parquet]",
            ) from exc

    @staticmethod
    def _load_hdf5(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load an HDF5 file."""
        try:
            import tables  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'tables' for HDF5 support: pip install tables",
            ) from exc

        key = kwargs.pop("key", None)
        if key:
            return pd.read_hdf(source, key=key, **kwargs)

        # If no key specified, use the first key
        with pd.HDFStore(source, mode="r") as store:
            keys = store.keys()
            if not keys:
                raise DataLoadError(source, reason="No datasets found in HDF5 file.")
            if len(keys) > 1:
                logger.warning(
                    "HDF5 contains %d keys: %s — using first key '%s'.",
                    len(keys),
                    keys,
                    keys[0],
                )
            return pd.read_hdf(source, key=keys[0], **kwargs)

    @staticmethod
    def _load_pickle(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a Pickle file.

        Warning:
            Only use pickle with trusted sources.
        """
        logger.warning("Loading pickle: verify this is a trusted source — %s", source)
        return pd.read_pickle(source, **kwargs)

    # ================================================================
    #  Statistical package formats
    # ================================================================

    @staticmethod
    def _load_sas(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a SAS data file (.sas7bdat)."""
        kwargs.setdefault("format", "sas7bdat")
        return pd.read_sas(source, **kwargs)

    @staticmethod
    def _load_sas_xport(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a SAS Transport file (.xpt)."""
        kwargs.setdefault("format", "xport")
        return pd.read_sas(source, **kwargs)

    @staticmethod
    def _load_stata(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a Stata file (.dta)."""
        return pd.read_stata(source, **kwargs)

    @staticmethod
    def _load_spss(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load an SPSS file (.sav, .zsav, .por)."""
        try:
            import pyreadstat  # noqa: F401
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'pyreadstat' for SPSS support: pip install pyreadstat",
            ) from exc
        return pd.read_spss(source, **kwargs)

    # ================================================================
    #  Databases
    # ================================================================

    @staticmethod
    def _load_sqlite(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a table from a SQLite database.

        Args:
            source: .db / .sqlite file path.
            **kwargs:
                table (str): Table name to load. Defaults to the first table.
                query (str): Direct SQL query. Takes precedence over ``table``.
        """
        import sqlite3

        table = kwargs.pop("table", None)
        query = kwargs.pop("query", None)
        conn = sqlite3.connect(source)

        try:
            if query:
                return pd.read_sql_query(query, conn, **kwargs)

            # Query table list
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()

            if not tables:
                raise DataLoadError(source, reason="No tables found in SQLite database.")

            if table is None:
                table = tables[0]
                if len(tables) > 1:
                    logger.warning(
                        "SQLite contains %d tables: %s — using '%s'.",
                        len(tables),
                        tables,
                        table,
                    )

            if table not in tables:
                raise DataLoadError(
                    source,
                    reason=f"Table '{table}' not found. Available: {tables}",
                )

            return pd.read_sql_query(f'SELECT * FROM "{table}"', conn, **kwargs)
        finally:
            conn.close()

    @staticmethod
    def _load_duckdb(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a table from a DuckDB database."""
        try:
            import duckdb
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'duckdb' for DuckDB support: pip install duckdb",
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
                raise DataLoadError(source, reason="No tables found in DuckDB database.")

            if table is None:
                table = table_names[0]
                if len(table_names) > 1:
                    logger.warning(
                        "DuckDB contains %d tables: %s — using '%s'.",
                        len(table_names),
                        table_names,
                        table,
                    )

            return conn.execute(f'SELECT * FROM "{table}"').fetchdf()
        finally:
            conn.close()

    # ================================================================
    #  Markup (XML / HTML)
    # ================================================================

    @staticmethod
    def _load_xml(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load an XML file."""
        try:
            import lxml  # noqa: F401
        except ImportError:
            logger.info("lxml not installed — using built-in etree parser.")
            kwargs.setdefault("parser", "etree")
        return pd.read_xml(source, **kwargs)

    @staticmethod
    def _load_html(source: str, **kwargs: Any) -> pd.DataFrame:
        """Extract tables from an HTML file.

        Returns the largest table if multiple tables are found.
        """
        try:
            import lxml  # noqa: F401
        except ImportError:
            logger.info("lxml not installed — using bs4 (html.parser).")
            kwargs.setdefault("flavor", "bs4")

        table_index = kwargs.pop("table_index", None)
        tables = pd.read_html(source, **kwargs)

        if not tables:
            raise DataLoadError(source, reason="No tables found in HTML file.")

        if table_index is not None:
            if table_index >= len(tables):
                raise DataLoadError(
                    source,
                    reason=f"table_index={table_index} out of range (total {len(tables)} tables)",
                )
            return tables[table_index]

        # Select the largest table
        if len(tables) > 1:
            sizes = [(i, len(t) * len(t.columns)) for i, t in enumerate(tables)]
            best_idx = max(sizes, key=lambda x: x[1])[0]
            logger.warning(
                "Found %d tables in HTML — using largest table #%d.",
                len(tables),
                best_idx,
            )
            return tables[best_idx]

        return tables[0]

    # ================================================================
    #  URL (remote files)
    # ================================================================

    def _load_url_auto(self, source: str, **kwargs: Any) -> pd.DataFrame:
        """Download and load a file from a URL.

        Infers format by analyzing Content-Type header and URL path.
        """
        import tempfile
        from urllib.parse import urlparse
        from urllib.request import urlopen, Request

        logger.info("Starting URL download: %s", source)

        req = Request(source, headers={"User-Agent": "f2a/0.1"})
        with urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            data = resp.read()

        # Infer format from Content-Type
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
            # Check URL path extension
            from f2a.utils.validators import SUPPORTED_EXTENSIONS

            path_ext = Path(urlparse(source).path).suffix.lower()
            detected_type = SUPPORTED_EXTENSIONS.get(path_ext, "csv")

        # Save to temp file and re-load with the appropriate loader
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

        logger.info("URL data → temp file (%s): %s", detected_type, tmp_path)

        method_name = self._LOADER_REGISTRY.get(detected_type)
        if method_name and hasattr(self, method_name):
            try:
                return getattr(self, method_name)(tmp_path, **kwargs)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # Default: try as CSV
        try:
            return self._load_csv(tmp_path, **kwargs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ================================================================
    #  HuggingFace Datasets
    # ================================================================

    @staticmethod
    def _load_huggingface(source: str, **kwargs: Any) -> pd.DataFrame:
        """Load a HuggingFace dataset.

        When neither ``config`` nor ``split`` is specified, all available
        configs × splits are discovered and concatenated into a single
        DataFrame with extra ``__subset__`` and ``__split__`` columns so the
        caller can distinguish each partition.

        To load only one specific partition, pass ``config`` and/or
        ``split`` explicitly.
        """
        try:
            from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
        except ImportError as exc:
            raise DataLoadError(
                source,
                reason="Install 'datasets' for HuggingFace support: pip install f2a[hf]",
            ) from exc

        # Extract dataset name from various formats
        dataset_name = source

        # HuggingFace URL: https://huggingface.co/datasets/org/name[/viewer/config[/split]]
        hf_match = HF_URL_PATTERN.match(dataset_name)
        if hf_match:
            dataset_name = hf_match.group("dataset")
            # Extract config/split from /viewer/... path if present
            url_config = hf_match.group("config")
            url_split = hf_match.group("split")
            if url_config and "config" not in kwargs:
                kwargs["config"] = url_config
            if url_split and "split" not in kwargs:
                kwargs["split"] = url_split
        else:
            # hf:// or huggingface:// prefix
            for prefix in HF_PREFIXES:
                if dataset_name.startswith(prefix):
                    dataset_name = dataset_name[len(prefix) :]
                    break

        # Strip trailing slashes
        dataset_name = dataset_name.rstrip("/")

        config = kwargs.pop("config", None)
        split = kwargs.pop("split", None)

        # --- explicit single-partition mode ---
        if config is not None or split is not None:
            split = split or "train"
            try:
                if config:
                    ds = load_dataset(dataset_name, config, split=split, **kwargs)
                else:
                    ds = load_dataset(dataset_name, split=split, **kwargs)
                return ds.to_pandas()
            except Exception as exc:
                raise DataLoadError(source, reason=str(exc)) from exc

        # --- auto-discover all configs × splits ---
        try:
            configs = get_dataset_config_names(dataset_name)
        except Exception:
            configs = [None]

        if not configs:
            configs = [None]

        frames: list[pd.DataFrame] = []
        for cfg in configs:
            try:
                if cfg is not None:
                    splits = get_dataset_split_names(dataset_name, cfg)
                else:
                    splits = get_dataset_split_names(dataset_name)
            except Exception:
                splits = ["train"]

            for sp in splits:
                try:
                    if cfg is not None:
                        ds = load_dataset(dataset_name, cfg, split=sp, **kwargs)
                    else:
                        ds = load_dataset(dataset_name, split=sp, **kwargs)
                    df_part = ds.to_pandas()
                    df_part["__subset__"] = cfg or "default"
                    df_part["__split__"] = sp
                    frames.append(df_part)
                    logger.info(
                        "HF partition loaded: config=%s split=%s (%d rows)",
                        cfg or "default", sp, len(df_part),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load config=%s split=%s: %s",
                        cfg, sp, exc,
                    )

        if not frames:
            raise DataLoadError(source, reason="No loadable configs/splits found.")

        return pd.concat(frames, ignore_index=True)
