# f2a (File to Analysis) — 기술 설계 문서

> **버전**: 0.1.0 (초안)
> **작성일**: 2026-03-13
> **상태**: 설계 단계

---

## 1. 프로젝트 개요

**f2a**는 다양한 데이터 소스(로컬 파일, Hugging Face 데이터셋 등)를 입력받아
**기술 통계 분석(Descriptive Statistics)** 및 **시각화(Visualization)** 를 자동으로 수행하는 Python 라이브러리입니다.

### 1.1 핵심 목표
- **원클릭 분석**: 파일 경로나 HuggingFace 주소 하나만으로 전체 기술 통계 + 시각화 수행
- **다양한 입력 지원**: CSV, JSON, Parquet, Excel, TSV, Hugging Face `datasets`
- **풍부한 통계**: 요약 통계, 분포 분석, 상관 분석, 결측치 분석
- **자동 시각화**: 히스토그램, 박스플롯, 상관 히트맵, 결측치 맵 등
- **리포트 생성**: 분석 결과를 HTML 리포트로 자동 출력

### 1.2 사용 시나리오

```python
import f2a

# 로컬 파일 분석
report = f2a.analyze("data/sales.csv")
report.show()            # 콘솔 요약 출력
report.to_html("out/")  # HTML 리포트 저장

# Hugging Face 데이터셋 분석
report = f2a.analyze("hf://imdb")
report.show()

# 세부 접근
report.stats.summary()        # 요약 통계 DataFrame
report.stats.correlation()    # 상관행렬
report.viz.plot_distributions()  # 분포 시각화
```

---

## 2. 아키텍처

### 2.1 계층 구조

```
┌─────────────────────────────────────────────┐
│                  Public API                 │
│         f2a.analyze() / f2a.load()          │
├─────────────────────────────────────────────┤
│               Core Orchestrator             │
│           Analyzer (파이프라인 제어)           │
├──────────┬──────────┬──────────┬────────────┤
│  Loader  │  Stats   │   Viz    │  Reporter  │
│ 데이터로딩 │ 통계분석  │  시각화   │ 리포트 생성 │
├──────────┴──────────┴──────────┴────────────┤
│                  Utilities                  │
│        타입 추론 · 유효성 검증 · 로깅          │
└─────────────────────────────────────────────┘
```

### 2.2 디렉토리 구조

```
f2a/
├── pyproject.toml            # 빌드 설정 (PEP 621)
├── README.md                 # 프로젝트 소개
├── PLAN.md                   # 이 문서
├── LICENSE                   # MIT License
│
├── src/
│   └── f2a/
│       ├── __init__.py       # Public API 노출
│       ├── _version.py       # 버전 관리
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── loader.py     # 파일/HF 데이터 로딩
│       │   ├── analyzer.py   # 분석 오케스트레이터
│       │   └── schema.py     # 컬럼 타입 추론 & 스키마
│       │
│       ├── stats/
│       │   ├── __init__.py
│       │   ├── descriptive.py    # 기술 통계 (평균, 중앙값, 분산 등)
│       │   ├── distribution.py   # 분포 분석 (왜도, 첨도, 정규성)
│       │   ├── correlation.py    # 상관 분석
│       │   └── missing.py        # 결측치 분석
│       │
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── theme.py          # 시각화 테마/스타일
│       │   ├── plots.py          # 기본 플롯 (히스토그램, 바, 박스)
│       │   ├── dist_plots.py     # 분포 시각화
│       │   ├── corr_plots.py     # 상관관계 시각화
│       │   └── missing_plots.py  # 결측치 시각화
│       │
│       ├── report/
│       │   ├── __init__.py
│       │   ├── generator.py      # 리포트 생성 엔진
│       │   └── templates/        # HTML 템플릿
│       │       └── base.html
│       │
│       └── utils/
│           ├── __init__.py
│           ├── type_inference.py  # 데이터 타입 자동 추론
│           ├── validators.py      # 입력 유효성 검증
│           └── logging.py         # 로깅 설정
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures
│   ├── test_loader.py
│   ├── test_descriptive.py
│   ├── test_correlation.py
│   ├── test_viz.py
│   └── test_report.py
│
└── examples/
    ├── quickstart.py         # 빠른 시작 예제
    └── huggingface_demo.py   # HF 데이터셋 예제
```

---

## 3. 핵심 모듈 설계

### 3.1 Loader (`core/loader.py`)

데이터 소스를 자동 감지하고 `pandas.DataFrame`으로 통일 변환합니다.

| 입력 형태 | 감지 방법 | 변환 방식 |
|---|---|---|
| CSV / TSV | 확장자 `.csv`, `.tsv` | `pd.read_csv()` |
| JSON / JSONL | 확장자 `.json`, `.jsonl` | `pd.read_json()` |
| Parquet | 확장자 `.parquet` | `pd.read_parquet()` |
| Excel | 확장자 `.xlsx`, `.xls` | `pd.read_excel()` |
| HuggingFace | `hf://` 프리픽스 또는 `org/dataset` 패턴 | `datasets.load_dataset()` → `.to_pandas()` |

**핵심 인터페이스:**
```python
class DataLoader:
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """소스 문자열을 분석하여 적절한 로더로 라우팅"""

    def _detect_source_type(self, source: str) -> SourceType:
        """소스 타입 자동 감지"""
```

### 3.2 Stats (`stats/`)

#### 3.2.1 기술 통계 (`descriptive.py`)

| 통계량 | 수치형 | 범주형 |
|---|---|---|
| count / unique | ✅ | ✅ |
| mean / median | ✅ | — |
| std / variance | ✅ | — |
| min / max / range | ✅ | — |
| Q1, Q3, IQR | ✅ | — |
| top / freq | — | ✅ |
| mode | ✅ | ✅ |

#### 3.2.2 분포 분석 (`distribution.py`)

- **왜도(Skewness)** & **첨도(Kurtosis)**
- **정규성 검정**: Shapiro-Wilk (n ≤ 5000), D'Agostino-Pearson
- **분위수 테이블**: 5%, 10%, 25%, 50%, 75%, 90%, 95%

#### 3.2.3 상관 분석 (`correlation.py`)

- **Pearson** 상관계수 (수치-수치)
- **Spearman** 순위 상관 (수치-수치, 비선형)
- **Cramér's V** (범주-범주)
- 다중공선성 경고 (|r| > 0.9)

#### 3.2.4 결측치 분석 (`missing.py`)

- 컬럼별 결측 비율
- 결측 패턴 분석 (MCAR / MAR 힌트)
- 행 단위 결측 분포

### 3.3 Viz (`viz/`)

| 차트 종류 | 대상 | 모듈 |
|---|---|---|
| 히스토그램 + KDE | 수치형 컬럼 | `dist_plots.py` |
| 박스플롯 | 수치형 컬럼 | `plots.py` |
| 바 차트 (빈도) | 범주형 컬럼 | `plots.py` |
| 상관 히트맵 | 수치형 컬럼 쌍 | `corr_plots.py` |
| 페어플롯 | 수치형 상위 N개 | `corr_plots.py` |
| 결측치 매트릭스 | 전체 | `missing_plots.py` |
| 바이올린 플롯 | 수치형 컬럼 | `dist_plots.py` |

**시각화 테마**: `viz/theme.py`에서 통일 스타일 관리 (컬러 팔레트, 폰트 크기 등)

### 3.4 Report (`report/`)

분석 결과를 종합한 HTML 리포트를 생성합니다.

**리포트 구성:**
1. **개요 섹션**: 데이터셋 이름, 행/열 수, 메모리 사용량
2. **변수 요약**: 컬럼별 타입, 결측률, 주요 통계
3. **분포 섹션**: 각 컬럼의 분포 시각화
4. **상관 섹션**: 상관 히트맵 + 주요 상관 쌍
5. **결측치 섹션**: 결측 패턴 시각화
6. **경고 섹션**: 이상값, 높은 상관, 높은 결측 등

---

## 4. 데이터 흐름

```
Input (파일 경로 / HF 주소)
       │
       ▼
  ┌─────────┐
  │  Loader  │ ──→ pd.DataFrame
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │  Schema  │ ──→ 컬럼 타입 추론 (수치/범주/텍스트/일시)
  └────┬─────┘
       │
       ├──→ Stats.descriptive()  ──→ StatResult
       ├──→ Stats.distribution() ──→ StatResult
       ├──→ Stats.correlation()  ──→ StatResult
       └──→ Stats.missing()      ──→ StatResult
              │
              ▼
       ┌────────────┐
       │  Viz Engine │ ──→ matplotlib Figure 객체들
       └─────┬──────┘
             │
             ▼
       ┌───────────┐
       │  Reporter  │ ──→ AnalysisReport
       └───────────┘
             │
             ├──→ .show()       (콘솔 출력)
             ├──→ .to_html()    (HTML 파일)
             └──→ .to_dict()    (프로그래밍 접근)
```

---

## 5. 의존성

### 5.1 필수 (Core)

| 패키지 | 버전 | 용도 |
|---|---|---|
| `pandas` | ≥ 2.0 | 데이터프레임 핵심 |
| `numpy` | ≥ 1.24 | 수치 연산 |
| `matplotlib` | ≥ 3.7 | 기본 시각화 |
| `seaborn` | ≥ 0.13 | 통계 시각화 |
| `scipy` | ≥ 1.11 | 통계 검정 |

### 5.2 선택 (Optional)

| 패키지 | 용도 | extras 이름 |
|---|---|---|
| `datasets` | HuggingFace 데이터셋 로딩 | `[hf]` |
| `openpyxl` | Excel 파일 지원 | `[excel]` |
| `pyarrow` | Parquet 파일 지원 | `[parquet]` |
| `rich` | 콘솔 출력 포매팅 | `[rich]` |
| `jinja2` | HTML 리포트 템플릿 | `[report]` |

### 5.3 설치 명령어

```bash
# 기본 설치
pip install f2a

# HuggingFace 지원 포함
pip install f2a[hf]

# 전체 기능
pip install f2a[all]
```

---

## 6. 개발 로드맵

### Phase 1 — 기초 (v0.1.0) ← **현재**
- [x] 프로젝트 구조 설정 (pyproject.toml, 디렉토리)
- [x] 기본 Loader (CSV, JSON)
- [x] 기술 통계 모듈 (descriptive.py)
- [x] 기본 시각화 (히스토그램, 박스플롯)
- [x] 콘솔 출력 (show)

### Phase 2 — 확장 (v0.2.0)
- [ ] HuggingFace 데이터셋 로더
- [ ] 상관 분석 & 히트맵
- [ ] 결측치 분석 & 시각화
- [ ] HTML 리포트 생성

### Phase 3 — 고도화 (v0.3.0)
- [ ] 분포 분석 (정규성 검정 등)
- [ ] 대용량 데이터 지원 (청크 로딩)
- [ ] 인터랙티브 시각화 (plotly 옵션)
- [ ] CLI 인터페이스

### Phase 4 — 안정화 (v1.0.0)
- [ ] API 안정화 & 문서화
- [ ] 종합 테스트 커버리지 > 80%
- [ ] PyPI 배포
- [ ] 튜토리얼 & 예제 노트북

---

## 7. 코딩 컨벤션

- **Python**: 3.10+
- **스타일**: PEP 8, Black 포매터, isort
- **타입 힌트**: 모든 public API에 필수
- **독스트링**: Google style
- **테스트**: pytest, 단위 테스트 원칙
- **린팅**: ruff

---

## 8. 핵심 클래스 설계

### 8.1 AnalysisReport

```python
@dataclass
class AnalysisReport:
    """분석 결과를 담는 최상위 컨테이너"""
    dataset_name: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    figures: dict[str, Figure]
    warnings: list[str]

    def show(self) -> None: ...
    def to_html(self, output_dir: str) -> Path: ...
    def to_dict(self) -> dict: ...
```

### 8.2 StatsResult

```python
@dataclass
class StatsResult:
    """통계 분석 결과 컨테이너"""
    summary: pd.DataFrame          # 요약 통계
    correlation_matrix: pd.DataFrame  # 상관행렬
    missing_info: pd.DataFrame     # 결측치 정보
    distribution_info: pd.DataFrame   # 분포 정보

    def get_numeric_summary(self) -> pd.DataFrame: ...
    def get_categorical_summary(self) -> pd.DataFrame: ...
```

---

## 9. 에러 처리 전략

| 상황 | 처리 방식 |
|---|---|
| 파일 미존재 | `FileNotFoundError` with 명확한 메시지 |
| 지원하지 않는 포맷 | `UnsupportedFormatError` (커스텀) |
| HF 데이터셋 로딩 실패 | `DataLoadError` (커스텀) + 원인 체이닝 |
| 빈 데이터셋 | `EmptyDataError` (커스텀) |
| 수치 컬럼 없음 | 경고 로그 + 해당 분석 스킵 |

---

*이 문서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
