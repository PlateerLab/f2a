"""입력 유효성 검증 유틸리티."""

from __future__ import annotations

import re
from pathlib import Path

from f2a.utils.exceptions import UnsupportedFormatError

# ── 지원 확장자 → 소스 타입 매핑 ────────────────────────────
# 새로운 포맷 추가 시 여기에만 등록하면 자동 라우팅됩니다.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    # CSV / 구분자 텍스트
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "delimited",  # 구분자 자동 감지
    ".dat": "delimited",
    ".tab": "tsv",
    # JSON 계열
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    # 스프레드시트
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".xlsb": "excel",
    ".ods": "ods",
    # 바이너리 / 컬럼나 포맷
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
    # 통계 패키지
    ".sas7bdat": "sas",
    ".xpt": "sas_xport",
    ".dta": "stata",
    ".sav": "spss",
    ".zsav": "spss",
    ".por": "spss",
    # 데이터베이스
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".ddb": "duckdb",
    ".duckdb": "duckdb",
    # 마크업 / 구조적 텍스트
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    # 고정 폭
    ".fwf": "fwf",
}

HF_PREFIXES = ("hf://", "huggingface://")
URL_PREFIXES = ("http://", "https://", "ftp://")


def detect_source_type(source: str) -> str:
    """소스 문자열로부터 데이터 소스 타입을 감지합니다.

    감지 우선순위:
        1. URL 프리픽스 (http/https/ftp)
        2. HuggingFace 프리픽스 (hf://, huggingface://)
        3. HuggingFace org/dataset 패턴
        4. 파일 확장자 매칭
        5. 멀티 확장자 매칭 (예: .sas7bdat → 마지막 `.` 이후와 전체 매칭)
        6. 콘텐츠 스니핑 (파일이 존재하는 경우)

    Args:
        source: 파일 경로, URL 또는 HuggingFace 주소.

    Returns:
        소스 타입 문자열 (``"csv"``, ``"json"``, ``"hf"``, ``"url"`` 등).

    Raises:
        UnsupportedFormatError: 지원하지 않는 포맷인 경우.
    """
    # 1. URL 감지
    for prefix in URL_PREFIXES:
        if source.lower().startswith(prefix):
            return _detect_url_type(source)

    # 2. HuggingFace 주소 감지
    for prefix in HF_PREFIXES:
        if source.startswith(prefix):
            return "hf"

    # 3. org/dataset 패턴 감지 (슬래시 포함, 확장자 없음)
    if "/" in source and not Path(source).suffix:
        parts = source.split("/")
        if len(parts) == 2 and all(
            re.match(r"^[a-zA-Z0-9_-]+$", part) for part in parts
        ):
            return "hf"

    # 4. 파일 확장자 기반 감지
    path = Path(source)
    ext = path.suffix.lower()

    # 멀티 확장자 처리 (.tar.gz, .sas7bdat 등)
    full_suffixes = "".join(path.suffixes).lower()
    if full_suffixes in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[full_suffixes]

    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    # 5. 파일이 존재하면 콘텐츠 스니핑 시도
    if path.exists() and path.is_file():
        sniffed = _sniff_content(path)
        if sniffed:
            return sniffed

    raise UnsupportedFormatError(source, detected=ext if ext else None)


def _detect_url_type(url: str) -> str:
    """URL에서 파일 타입을 추출합니다.

    URL 경로의 확장자를 확인하고, 없으면 ``"url_csv"`` 로 기본 처리합니다.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path
    ext = Path(path).suffix.lower()

    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    # 확장자가 없으면 URL로 표시하여 CSV 시도 (가장 흔한 경우)
    return "url_auto"


def _sniff_content(path: Path, peek_bytes: int = 8192) -> str | None:
    """파일의 처음 몇 바이트를 읽어 포맷을 추측합니다.

    Args:
        path: 파일 경로.
        peek_bytes: 읽을 바이트 수.

    Returns:
        감지된 소스 타입 문자열 또는 None.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(peek_bytes)
    except (OSError, PermissionError):
        return None

    # ── 바이너리 매직 넘버 ──
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

    # Pickle: 여러 프로토콜 매직
    if header[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return "pickle"

    # Excel XLSX (ZIP): "PK\x03\x04"
    if header[:4] == b"PK\x03\x04":
        # ZIP 파일 — XLSX일 수 있음
        if b"xl/" in header or b"[Content_Types].xml" in header:
            return "excel"
        return None

    # Excel XLS (OLE2): "\xd0\xcf\x11\xe0"
    if header[:4] == b"\xd0\xcf\x11\xe0":
        return "excel"

    # ── 텍스트 기반 스니핑 ──
    try:
        text = header.decode("utf-8", errors="replace")
    except Exception:
        return None

    text_stripped = text.strip()

    # JSON
    if text_stripped.startswith(("{", "[")):
        # JSONL: 여러 줄의 JSON
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

    # CSV vs TSV — 구분자 감지
    if "\t" in text_stripped:
        tab_count = text_stripped.count("\t")
        comma_count = text_stripped.count(",")
        if tab_count > comma_count:
            return "tsv"

    if "," in text_stripped:
        return "csv"

    # 기본적으로 구분자 텍스트로 시도
    if "\n" in text_stripped and len(text_stripped.split("\n")) > 1:
        return "delimited"

    return None


def get_supported_formats() -> dict[str, list[str]]:
    """지원하는 포맷과 확장자 목록을 반환합니다.

    Returns:
        포맷 이름 → 확장자 리스트 매핑.
    """
    result: dict[str, list[str]] = {}
    for ext, fmt in SUPPORTED_EXTENSIONS.items():
        result.setdefault(fmt, []).append(ext)
    result["hf"] = ["hf://...", "org/dataset"]
    result["url"] = ["http://...", "https://..."]
    return result


def validate_source(source: str) -> str:
    """소스 문자열을 검증하고 정규화합니다.

    Args:
        source: 입력 소스 문자열.

    Returns:
        정규화된 소스 문자열.

    Raises:
        ValueError: 빈 문자열인 경우.
    """
    if not source or not source.strip():
        raise ValueError("소스 문자열이 비어 있습니다.")
    return source.strip()
