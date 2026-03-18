"""Input validation utilities."""

from __future__ import annotations

import re
from pathlib import Path

from f2a.utils.exceptions import UnsupportedFormatError

# ── Supported extensions → source type mapping ────────
# Register new formats here; they will be auto-routed.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    # CSV / delimited text
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "delimited",  # auto-detect delimiter
    ".dat": "delimited",
    ".tab": "tsv",
    # JSON family
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    # Spreadsheets
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".xlsb": "excel",
    ".ods": "ods",
    # Binary / columnar formats
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".ftr": "feather",
    ".arrow": "arrow_ipc",
    ".ipc": "arrow_ipc",
    ".orc": "orc",
    ".hdf": "hdf5",
    ".hdf5": "hdf5",
    ".h5": "hdf5",
    ".pkl": "pickle",
    ".pickle": "pickle",
    # Statistical packages
    ".sas7bdat": "sas",
    ".xpt": "sas_xport",
    ".dta": "stata",
    ".sav": "spss",
    ".zsav": "spss",
    ".por": "spss",
    # Databases
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".ddb": "duckdb",
    ".duckdb": "duckdb",
    # Markup / structured text
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    # Fixed-width
    ".fwf": "fwf",
}

HF_PREFIXES = ("hf://", "huggingface://")
HF_URL_PATTERN = re.compile(
    r"^https?://huggingface\.co/datasets/"
    r"(?P<dataset>[^/?#]+(?:/[^/?#]+)?)"
    r"(?:/viewer(?:/(?P<config>[^/?#]+))?(?:/(?P<split>[^/?#]+))?)?",
    re.IGNORECASE,
)
URL_PREFIXES = ("http://", "https://", "ftp://")


def detect_source_type(source: str) -> str:
    """Detect data source type from a source string.

    Detection priority:
        1. HuggingFace URL (https://huggingface.co/datasets/...)
        2. URL prefix (http/https/ftp)
        3. HuggingFace prefix (hf://, huggingface://)
        4. HuggingFace org/dataset pattern
        5. File extension matching
        6. Multi-extension matching (e.g., .sas7bdat)
        7. Content sniffing (if file exists)

    Args:
        source: File path, URL, or HuggingFace address.

    Returns:
        Source type string (``"csv"``, ``"json"``, ``"hf"``, ``"url"``, etc.).

    Raises:
        UnsupportedFormatError: If the format is not supported.
    """
    # 1. HuggingFace URL detection (before generic URL handling)
    if HF_URL_PATTERN.match(source):
        return "hf"

    # 2. URL detection
    for prefix in URL_PREFIXES:
        if source.lower().startswith(prefix):
            return _detect_url_type(source)

    # 3. HuggingFace prefix detection (hf://, huggingface://)
    for prefix in HF_PREFIXES:
        if source.startswith(prefix):
            return "hf"

    # 4. org/dataset pattern detection (contains slash, no extension)
    if "/" in source and not Path(source).suffix:
        parts = source.split("/")
        if len(parts) == 2 and all(
            re.match(r"^[a-zA-Z0-9_-]+$", part) for part in parts
        ):
            return "hf"

    # 5. File extension-based detection
    path = Path(source)
    ext = path.suffix.lower()

    # Multi-extension handling (.tar.gz, .sas7bdat, etc.)
    full_suffixes = "".join(path.suffixes).lower()
    if full_suffixes in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[full_suffixes]

    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    # 6. Attempt content sniffing if file exists
    if path.exists() and path.is_file():
        sniffed = _sniff_content(path)
        if sniffed:
            return sniffed

    raise UnsupportedFormatError(source, detected=ext if ext else None)


def _detect_url_type(url: str) -> str:
    """Extract file type from URL.

    Check the URL path extension; defaults to ``"url_auto"`` if none found.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path
    ext = Path(path).suffix.lower()

    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    # No extension found — mark as URL for auto-detection
    return "url_auto"


def _sniff_content(path: Path, peek_bytes: int = 8192) -> str | None:
    """Read the first few bytes of a file to guess its format.

    Args:
        path: File path.
        peek_bytes: Number of bytes to read.

    Returns:
        Detected source type string, or None.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(peek_bytes)
    except (OSError, PermissionError):
        return None

    # ── Binary magic numbers ──
    # Parquet: "PAR1"
    if header[:4] == b"PAR1":
        return "parquet"

    # Apache Arrow IPC: "ARROW1"
    if header[:6] == b"ARROW1":
        return "arrow_ipc"

    # ORC: "ORC"
    if header[:3] == b"ORC":
        return "orc"

    # HDF5: "\x89HDF\r\n\x1a\n"
    if header[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"

    # Feather (Arrow IPC v2): "ARROW1" or FEA1
    if header[:4] == b"FEA1":
        return "feather"

    # SQLite: "SQLite format 3\x00"
    if header[:16] == b"SQLite format 3\x00":
        return "sqlite"

    # Pickle: various protocol magic bytes
    if header[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return "pickle"

    # Excel XLSX (ZIP): "PK\x03\x04"
    if header[:4] == b"PK\x03\x04":
        # ZIP file — could be XLSX
        if b"xl/" in header or b"[Content_Types].xml" in header:
            return "excel"
        return None

    # Excel XLS (OLE2): "\xd0\xcf\x11\xe0"
    if header[:4] == b"\xd0\xcf\x11\xe0":
        return "excel"

    # ── Text-based sniffing ──
    try:
        text = header.decode("utf-8", errors="replace")
    except Exception:
        return None

    text_stripped = text.strip()

    # JSON
    if text_stripped.startswith(("{", "[")):
        # JSONL: multi-line JSON objects
        lines = text_stripped.split("\n", 5)
        if len(lines) > 1 and all(
            line.strip().startswith("{") for line in lines[:3] if line.strip()
        ):
            return "jsonl"
        return "json"

    # XML / HTML
    if text_stripped.startswith("<?xml") or text_stripped.startswith("<"):
        if "<html" in text_stripped.lower() or "<table" in text_stripped.lower():
            return "html"
        return "xml"

    # CSV vs TSV — delimiter detection
    if "\t" in text_stripped:
        tab_count = text_stripped.count("\t")
        comma_count = text_stripped.count(",")
        if tab_count > comma_count:
            return "tsv"

    if "," in text_stripped:
        return "csv"

    # Default: try as delimited text
    if "\n" in text_stripped and len(text_stripped.split("\n")) > 1:
        return "delimited"

    return None


def get_supported_formats() -> dict[str, list[str]]:
    """Return supported formats and their file extensions.

    Returns:
        Format name → extension list mapping.
    """
    result: dict[str, list[str]] = {}
    for ext, fmt in SUPPORTED_EXTENSIONS.items():
        result.setdefault(fmt, []).append(ext)
    result["hf"] = ["hf://...", "org/dataset", "https://huggingface.co/datasets/..."]
    result["url"] = ["http://...", "https://..."]
    return result


def validate_source(source: str) -> str:
    """Validate and normalize a source string.

    Args:
        source: Input source string.

    Returns:
        Normalized source string.

    Raises:
        ValueError: If the source string is empty.
    """
    if not source or not source.strip():
        raise ValueError("Source string is empty.")
    return source.strip()
