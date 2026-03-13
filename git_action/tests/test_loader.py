"""DataLoader tests — auto-detection and loading of various formats."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f2a.core.loader import DataLoader
from f2a.utils.exceptions import DataLoadError, EmptyDataError, UnsupportedFormatError
from f2a.utils.validators import detect_source_type, get_supported_formats


# ── fixtures: test file creation for various formats ──────────────────

@pytest.fixture
def base_df() -> pd.DataFrame:
    """Base DataFrame used for all format tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(1, 51),
            "name": [f"item_{i}" for i in range(1, 51)],
            "value": np.random.normal(100, 20, 50).round(2),
            "category": np.random.choice(["A", "B", "C"], 50),
            "score": np.random.uniform(0, 100, 50).round(1),
        }
    )


@pytest.fixture
def csv_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.csv"
    base_df.to_csv(p, index=False)
    return p


@pytest.fixture
def tsv_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.tsv"
    base_df.to_csv(p, index=False, sep="\t")
    return p


@pytest.fixture
def json_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.json"
    base_df.to_json(p, orient="records", force_ascii=False)
    return p


@pytest.fixture
def jsonl_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.jsonl"
    base_df.to_json(p, orient="records", lines=True, force_ascii=False)
    return p


@pytest.fixture
def ndjson_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    """ndjson extension test."""
    p = tmp_path / "test.ndjson"
    base_df.to_json(p, orient="records", lines=True, force_ascii=False)
    return p


@pytest.fixture
def delimited_pipe_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    """Pipe (|) delimited text file."""
    p = tmp_path / "test.txt"
    base_df.to_csv(p, index=False, sep="|")
    return p


@pytest.fixture
def delimited_semicolon_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    """Semicolon (;) delimited dat file."""
    p = tmp_path / "test.dat"
    base_df.to_csv(p, index=False, sep=";")
    return p


@pytest.fixture
def nested_json_file(tmp_path: Path) -> Path:
    """Nested JSON file."""
    data = {
        "metadata": {"source": "test", "version": 1},
        "records": [
            {"id": 1, "name": "a", "val": 10},
            {"id": 2, "name": "b", "val": 20},
            {"id": 3, "name": "c", "val": 30},
        ],
    }
    p = tmp_path / "nested.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def sqlite_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.db"
    conn = sqlite3.connect(str(p))
    base_df.to_sql("main_table", conn, index=False)
    base_df.head(10).to_sql("small_table", conn, index=False)
    conn.close()
    return p


@pytest.fixture
def stata_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.dta"
    base_df.to_stata(p, write_index=False)
    return p


@pytest.fixture
def pickle_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.pkl"
    base_df.to_pickle(p)
    return p


@pytest.fixture
def html_file(tmp_path: Path, base_df: pd.DataFrame) -> Path:
    p = tmp_path / "test.html"
    html_content = f"""<html><body>
    <h1>Test</h1>
    <table><tr><td>ignore</td></tr></table>
    {base_df.to_html(index=False)}
    </body></html>"""
    p.write_text(html_content, encoding="utf-8")
    return p


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.xml"
    xml_content = """<?xml version="1.0"?>
<data>
  <row><id>1</id><name>alpha</name><value>10.5</value></row>
  <row><id>2</id><name>beta</name><value>20.3</value></row>
  <row><id>3</id><name>gamma</name><value>30.1</value></row>
</data>"""
    p.write_text(xml_content, encoding="utf-8")
    return p


@pytest.fixture
def cp949_csv_file(tmp_path: Path) -> Path:
    """CP949-encoded CSV (Korean)."""
    p = tmp_path / "korean.csv"
    df = pd.DataFrame({"이름": ["홍길동", "김철수"], "나이": [30, 25]})
    df.to_csv(p, index=False, encoding="cp949")
    return p


@pytest.fixture
def fwf_file(tmp_path: Path) -> Path:
    """Fixed-width file."""
    p = tmp_path / "test.fwf"
    content = """Name      Age  Score
Alice      28   95.5
Bob        34   87.3
Charlie    22   91.0
"""
    p.write_text(content, encoding="utf-8")
    return p


# ================================================================
#  Source type detection tests
# ================================================================


class TestDetectSourceType:
    """Source type detection tests."""

    # ── Extension-based ──
    def test_csv(self) -> None:
        assert detect_source_type("data.csv") == "csv"

    def test_tsv(self) -> None:
        assert detect_source_type("data.tsv") == "tsv"

    def test_tab(self) -> None:
        assert detect_source_type("data.tab") == "tsv"

    def test_txt(self) -> None:
        assert detect_source_type("data.txt") == "delimited"

    def test_dat(self) -> None:
        assert detect_source_type("data.dat") == "delimited"

    def test_json(self) -> None:
        assert detect_source_type("data.json") == "json"

    def test_jsonl(self) -> None:
        assert detect_source_type("data.jsonl") == "jsonl"

    def test_ndjson(self) -> None:
        assert detect_source_type("data.ndjson") == "jsonl"

    def test_parquet(self) -> None:
        assert detect_source_type("data.parquet") == "parquet"

    def test_pq(self) -> None:
        assert detect_source_type("data.pq") == "parquet"

    def test_excel_xlsx(self) -> None:
        assert detect_source_type("data.xlsx") == "excel"

    def test_excel_xls(self) -> None:
        assert detect_source_type("data.xls") == "excel"

    def test_excel_xlsm(self) -> None:
        assert detect_source_type("data.xlsm") == "excel"

    def test_excel_xlsb(self) -> None:
        assert detect_source_type("data.xlsb") == "excel"

    def test_ods(self) -> None:
        assert detect_source_type("data.ods") == "ods"

    def test_feather(self) -> None:
        assert detect_source_type("data.feather") == "feather"

    def test_arrow_ipc(self) -> None:
        assert detect_source_type("data.arrow") == "arrow_ipc"

    def test_orc(self) -> None:
        assert detect_source_type("data.orc") == "orc"

    def test_hdf5(self) -> None:
        assert detect_source_type("data.h5") == "hdf5"

    def test_hdf5_ext(self) -> None:
        assert detect_source_type("data.hdf5") == "hdf5"

    def test_pickle(self) -> None:
        assert detect_source_type("data.pkl") == "pickle"

    def test_pickle_ext(self) -> None:
        assert detect_source_type("data.pickle") == "pickle"

    def test_sas(self) -> None:
        assert detect_source_type("data.sas7bdat") == "sas"

    def test_sas_xport(self) -> None:
        assert detect_source_type("data.xpt") == "sas_xport"

    def test_stata(self) -> None:
        assert detect_source_type("data.dta") == "stata"

    def test_spss(self) -> None:
        assert detect_source_type("data.sav") == "spss"

    def test_sqlite(self) -> None:
        assert detect_source_type("data.db") == "sqlite"

    def test_sqlite3(self) -> None:
        assert detect_source_type("data.sqlite3") == "sqlite"

    def test_duckdb(self) -> None:
        assert detect_source_type("data.duckdb") == "duckdb"

    def test_xml(self) -> None:
        assert detect_source_type("data.xml") == "xml"

    def test_html(self) -> None:
        assert detect_source_type("data.html") == "html"

    def test_htm(self) -> None:
        assert detect_source_type("data.htm") == "html"

    def test_fwf(self) -> None:
        assert detect_source_type("data.fwf") == "fwf"

    # ── HuggingFace ──
    def test_hf_prefix(self) -> None:
        assert detect_source_type("hf://imdb") == "hf"

    def test_hf_huggingface_prefix(self) -> None:
        assert detect_source_type("huggingface://squad") == "hf"

    def test_hf_org_pattern(self) -> None:
        assert detect_source_type("openai/gsm8k") == "hf"

    # ── URL ──
    def test_url_csv(self) -> None:
        result = detect_source_type("https://example.com/data.csv")
        assert result == "csv"

    def test_url_json(self) -> None:
        result = detect_source_type("https://example.com/api/data.json")
        assert result == "json"

    def test_url_no_ext(self) -> None:
        result = detect_source_type("https://example.com/api/data")
        assert result == "url_auto"

    # ── Errors ──
    def test_unsupported(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            detect_source_type("data.xyz")

    # ── Utilities ──
    def test_supported_formats_returns_dict(self) -> None:
        result = get_supported_formats()
        assert isinstance(result, dict)
        assert "csv" in result
        assert "hf" in result
        assert "url" in result

    def test_loader_supported_formats(self) -> None:
        formats = DataLoader.supported_formats()
        assert "csv" in formats
        assert "parquet" in formats
        assert "sqlite" in formats
        assert "hf" in formats


# ================================================================
#  Content sniffing tests
# ================================================================


class TestContentSniffing:
    """Content-based detection tests for files without extensions."""

    def test_sniff_csv_content(self, tmp_path: Path) -> None:
        p = tmp_path / "noext"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
        assert detect_source_type(str(p)) == "csv"

    def test_sniff_tsv_content(self, tmp_path: Path) -> None:
        p = tmp_path / "noext_tsv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False, sep="\t")
        assert detect_source_type(str(p)) == "tsv"

    def test_sniff_json_content(self, tmp_path: Path) -> None:
        p = tmp_path / "noext_json"
        p.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
        assert detect_source_type(str(p)) == "json"

    def test_sniff_jsonl_content(self, tmp_path: Path) -> None:
        p = tmp_path / "noext_jsonl"
        p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
        assert detect_source_type(str(p)) == "jsonl"

    def test_sniff_xml_content(self, tmp_path: Path) -> None:
        p = tmp_path / "noext_xml"
        p.write_text('<?xml version="1.0"?><data><r><a>1</a></r></data>', encoding="utf-8")
        assert detect_source_type(str(p)) == "xml"

    def test_sniff_html_content(self, tmp_path: Path) -> None:
        p = tmp_path / "noext_html"
        p.write_text('<html><body><table><tr><td>1</td></tr></table></body></html>', encoding="utf-8")
        assert detect_source_type(str(p)) == "html"

    def test_sniff_sqlite_content(self, tmp_path: Path) -> None:
        p = tmp_path / "noext_db"
        conn = sqlite3.connect(str(p))
        pd.DataFrame({"x": [1]}).to_sql("t", conn, index=False)
        conn.close()
        assert detect_source_type(str(p)) == "sqlite"


# ================================================================
#  DataLoader — file loading tests
# ================================================================


class TestDataLoaderCSV:
    """CSV loading tests."""

    def test_load_csv(self, csv_file: Path) -> None:
        df = DataLoader().load(str(csv_file))
        assert len(df) == 50
        assert "id" in df.columns

    def test_load_csv_cp949(self, cp949_csv_file: Path) -> None:
        """CP949-encoded CSV auto-handling test."""
        df = DataLoader().load(str(cp949_csv_file))
        assert len(df) == 2
        assert "이름" in df.columns


class TestDataLoaderTSV:
    def test_load_tsv(self, tsv_file: Path) -> None:
        df = DataLoader().load(str(tsv_file))
        assert len(df) == 50


class TestDataLoaderJSON:
    def test_load_json(self, json_file: Path) -> None:
        df = DataLoader().load(str(json_file))
        assert len(df) == 50

    def test_load_jsonl(self, jsonl_file: Path) -> None:
        df = DataLoader().load(str(jsonl_file))
        assert len(df) == 50

    def test_load_ndjson(self, ndjson_file: Path) -> None:
        df = DataLoader().load(str(ndjson_file))
        assert len(df) == 50

    def test_load_nested_json(self, nested_json_file: Path) -> None:
        """Nested JSON auto-flatten test."""
        df = DataLoader().load(str(nested_json_file))
        assert len(df) == 3
        assert "id" in df.columns


class TestDataLoaderDelimited:
    def test_load_pipe_delimited(self, delimited_pipe_file: Path) -> None:
        """Pipe delimiter auto-detection test."""
        df = DataLoader().load(str(delimited_pipe_file))
        assert len(df) == 50
        assert len(df.columns) >= 5

    def test_load_semicolon_delimited(self, delimited_semicolon_file: Path) -> None:
        """Semicolon delimiter auto-detection test."""
        df = DataLoader().load(str(delimited_semicolon_file))
        assert len(df) == 50
        assert len(df.columns) >= 5

    def test_load_fwf(self, fwf_file: Path) -> None:
        df = DataLoader().load(str(fwf_file))
        assert len(df) >= 3


class TestDataLoaderSQLite:
    def test_load_sqlite_auto(self, sqlite_file: Path) -> None:
        """First table auto-selection test."""
        df = DataLoader().load(str(sqlite_file))
        assert len(df) == 50

    def test_load_sqlite_specific_table(self, sqlite_file: Path) -> None:
        df = DataLoader().load(str(sqlite_file), table="small_table")
        assert len(df) == 10

    def test_load_sqlite_query(self, sqlite_file: Path) -> None:
        df = DataLoader().load(
            str(sqlite_file), query="SELECT * FROM main_table WHERE value > 100"
        )
        assert len(df) > 0

    def test_load_sqlite_missing_table(self, sqlite_file: Path) -> None:
        with pytest.raises(DataLoadError, match="Table"):
            DataLoader().load(str(sqlite_file), table="nonexistent")


class TestDataLoaderStata:
    def test_load_stata(self, stata_file: Path) -> None:
        df = DataLoader().load(str(stata_file))
        assert len(df) == 50


class TestDataLoaderPickle:
    def test_load_pickle(self, pickle_file: Path) -> None:
        df = DataLoader().load(str(pickle_file))
        assert len(df) == 50


class TestDataLoaderHTML:
    def test_load_html_largest_table(self, html_file: Path) -> None:
        """HTML largest table auto-selection test."""
        df = DataLoader().load(str(html_file))
        assert len(df) == 50

    def test_load_html_specific_table(self, html_file: Path) -> None:
        df = DataLoader().load(str(html_file), table_index=1)
        assert len(df) == 50


class TestDataLoaderXML:
    def test_load_xml(self, xml_file: Path) -> None:
        df = DataLoader().load(str(xml_file))
        assert len(df) == 3
        assert "name" in df.columns


class TestDataLoaderErrors:
    def test_nonexistent_file(self) -> None:
        with pytest.raises(Exception):
            DataLoader().load("nonexistent_file.csv")

    def test_empty_source(self) -> None:
        with pytest.raises(ValueError):
            from f2a.utils.validators import validate_source
            validate_source("")

    def test_unsupported_format(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            DataLoader().load("file.xyz")
