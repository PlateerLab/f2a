# f2a — File to Analysis

> 데이터 소스를 입력하면 기술 통계 분석 및 시각화를 자동으로 수행하는 Python 라이브러리

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 설치

```bash
pip install f2a

# HuggingFace 데이터셋 지원 포함
pip install f2a[hf]

# 전체 기능
pip install f2a[all]
```

## 빠른 시작

```python
import f2a

# 로컬 CSV 파일 분석
report = f2a.analyze("data/sales.csv")
report.show()  # 콘솔에 요약 출력

# Hugging Face 데이터셋 분석
report = f2a.analyze("hf://imdb")
report.show()

# 세부 결과 접근
report.stats.summary        # 요약 통계 DataFrame
report.stats.correlation     # 상관행렬
report.viz.plot_distributions()  # 분포 시각화
```

## 지원 포맷

| 포맷 | 확장자 | 추가 설치 |
|---|---|---|
| CSV / TSV | `.csv`, `.tsv` | — |
| JSON / JSONL | `.json`, `.jsonl` | — |
| Parquet | `.parquet` | `pip install f2a[parquet]` |
| Excel | `.xlsx`, `.xls` | `pip install f2a[excel]` |
| HuggingFace | `hf://dataset_name` | `pip install f2a[hf]` |

## 분석 항목

- **기술 통계**: 평균, 중앙값, 표준편차, 분위수, 최빈값 등
- **분포 분석**: 왜도, 첨도, 정규성 검정
- **상관 분석**: Pearson, Spearman, Cramér's V
- **결측치 분석**: 결측 비율, 패턴 분석
- **시각화**: 히스토그램, 박스플롯, 상관 히트맵, 결측치 매트릭스

## 개발

```bash
git clone https://github.com/f2a/f2a.git
cd f2a
pip install -e ".[dev]"
pytest
```

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
