# f2a Enhancement Master Plan — v2.0 Masterpiece

> **Date**: 2026-03-16
> **Status**: Design Complete → Implementation Phase
> **Goal**: f2a를 단순 통계 덤프 도구에서 **자동 인사이트 생성 + 다차원 교차분석 + ML 준비도 평가** 를 갖춘 최고 수준의 EDA 엔진으로 고도화한다.

---

## 0. 현재 상태 진단 (Diagnosis)

### 0.1 현재 강점
| 영역 | 구현 상태 | 비고 |
|------|-----------|------|
| 데이터 로더 | ★★★★★ | 24+ 포맷, HuggingFace 다중 subset 자동 탐색 |
| 기초 통계 | ★★★★★ | 16개 기술통계, 4개 정규성 검정, 5종 상관분석 |
| 고급 통계 | ★★★★☆ | 7종 분포 피팅, 부분상관, MI 행렬, IF/LOF/Mahalanobis 등 |
| 시각화 | ★★★★☆ | 12종 Plotter, 30+ 차트 유형, base64 인라인 |
| HTML 리포트 | ★★★★☆ | 2-depth 탭, 드래그 스크롤, 툴팁, 모달, i18n 6개 언어 |
| 전처리 | ★★★★☆ | 상수/고결측/ID성/혼합타입/무한값 자동 탐지·정제 |

### 0.2 핵심 약점 (Gap Analysis)

| # | 약점 | 영향도 | 현재 상태 |
|---|------|--------|-----------|
| **G1** | **자동 인사이트 엔진 부재** | ★★★★★ | 보고서가 수치/차트만 나열 — "그래서 뭐가 중요한데?" 대답 불가 |
| **G2** | **교차분석(cross-dimensional) 부재** | ★★★★★ | 각 분석(상관/이상치/클러스터/분포)이 고립적 수행 — 연계 패턴 미탐지 |
| **G3** | **Data Profile 탭이 stub** | ★★★★☆ | 7개 숫자 카드만 렌더링, 컬럼별 프로파일/샘플/메모리 분석 없음 |
| **G4** | **Dim Reduction / Feature Insights / Stat Tests 차트 누락** | ★★★★☆ | 섹션 빌더에 chart_keys 미등록 → 생성된 Figure도 미표시 |
| **G5** | **통계 검정 시맨틱 오류** | ★★★☆☆ | Kruskal-Wallis가 범주형 그룹 변수 없이 수치 컬럼 간 비교 |
| **G6** | **다중검정 보정 없음** | ★★★☆☆ | 쌍별 검정(Levene, Mann-Whitney)에서 Bonferroni/FDR 미적용 |
| **G7** | **효과 크기(Effect Size) 미제공** | ★★★☆☆ | p-value만 보고, Cohen's d / η² / Cramér's V 없음 |
| **G8** | **컬럼 역할 자동 분류 없음** | ★★★☆☆ | ID/타겟/피처/시간/텍스트 역할 미추론 |
| **G9** | **ML 준비도 평가 없음** | ★★★☆☆ | "이 데이터가 ML에 바로 쓸 수 있나?" 판단 기능 없음 |
| **G10** | **Health Radar 차트 없음** | ★★☆☆☆ | 수평 바만 존재, 방사형 종합 대시보드 미구현 |

---

## 1. Enhancement 아키텍처 설계

### 1.1 신규 모듈 구조 (추가 파일)

```
f2a/
├── stats/
│   ├── insight_engine.py       ← [신규] 자동 인사이트 생성 엔진
│   ├── cross_analysis.py       ← [신규] 교차 분석 모듈
│   ├── column_role.py          ← [신규] 컬럼 역할 자동 분류
│   └── ml_readiness.py         ← [신규] ML 준비도 평가
│
├── viz/
│   ├── insight_plots.py        ← [신규] 인사이트 시각화
│   ├── cross_plots.py          ← [신규] 교차 분석 시각화
│   └── dimreduction_plots.py   ← [신규] 차원축소 전용 시각화
│
├── core/
│   └── analyzer.py             ← [수정] 새 모듈 통합 + VizResult 확장
│
└── report/
    └── generator.py            ← [수정] 인사이트 패널, 교차분석 섹션, 누락 차트 등록
```

### 1.2 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `core/config.py` | 새 분석 토글 추가 (insight_engine, cross_analysis, column_role, ml_readiness) |
| `core/analyzer.py` | `_compute_stats()` 확장, VizResult에 새 plot 메서드, `_compute_advanced_stats()` 확장 |
| `stats/statistical_tests.py` | Kruskal-Wallis 시맨틱 수정, 다중검정 보정 추가, 효과 크기 추가 |
| `stats/quality.py` | consistency() 성능 개선, 새 차원(timeliness, conformity) 추가 |
| `report/generator.py` | Data Profile 강화, 누락 차트 등록, 인사이트 패널, 교차분석 섹션, Health Radar |
| `report/i18n.py` | 새 섹션/인사이트 번역 키 추가 |

---

## 2. Enhancement 상세 설계

### Phase 1: 자동 인사이트 엔진 (Insight Engine) — `stats/insight_engine.py`

> **목적**: 모든 분석 결과를 종합하여 우선순위화된 자연어 인사이트를 자동 생성한다.

#### 2.1.1 인사이트 타입 분류

| 타입 | 아이콘 | 설명 | 예시 |
|------|--------|------|------|
| `FINDING` | 🔍 | 주목할 만한 데이터 패턴/사실 발견 | "column 'price'는 강한 오른쪽 꼬리(skew=2.3)를 가지며, log 변환으로 정규성이 크게 개선됩니다" |
| `WARNING` | ⚠️ | 데이터 품질/이상 경고 | "3개 컬럼 쌍에서 r>0.95의 다중공선성이 탐지되었습니다. VIF 기반 제거를 검토하세요" |
| `RECOMMENDATION` | 💡 | 데이터 전처리/모델링 제안 | "'age' 컬럼에 5.2%의 결측이 있습니다. 분포가 정규에 가까우므로 평균 대체가 적합합니다" |
| `OPPORTUNITY` | 🚀 | 활용 가능한 패턴/기회 | "K-Means(k=3)에서 뚜렷한 3개 군집이 형성됩니다. 군집별 특성 분석으로 세분화 전략 수립이 가능합니다" |

#### 2.1.2 인사이트 생성 규칙 (Rules Engine)

각 분석 모듈의 결과에서 다음 규칙들을 체계적으로 적용:

**분포 기반 인사이트:**
```python
class DistributionInsightRules:
    """분포 분석에서 인사이트를 추출하는 규칙 엔진."""

    rules = [
        # (조건 함수, 인사이트 생성 함수, 심각도/우선순위)
        (lambda col: abs(col.skewness) > 2.0,
         "극단적 비대칭: '{col}'의 skewness={val:.2f}. {transform} 변환 권장",
         "high"),
        (lambda col: col.kurtosis > 7.0,
         "극단적 첨도: '{col}'에 heavy tail ({val:.1f}). 이상치가 통계량을 왜곡할 수 있음",
         "high"),
        (lambda col: col.is_normal and col.cv < 0.1,
         "'{col}'는 정규분포이며 변동성이 매우 낮음 (CV={val:.3f}). 안정적 특성",
         "low"),
        (lambda col: not col.is_normal and col.best_fit != 'norm',
         "'{col}'는 {best_fit} 분포에 가장 적합 (AIC 기준). 파라미터 변환 시 유용",
         "medium"),
    ]
```

**상관 기반 인사이트:**
- 다중공선성 탐지 → VIF>10 또는 |r|>0.9인 쌍 식별 + 제거 후보 추천
- 비선형 의존성 → MI가 높으나 Pearson이 낮은 쌍 (MI/max(MI) > 0.5 & |r| < 0.3)
- 교란 변수 의심 → 편상관에서 크게 감소하는 쌍 (|partial_r - r| > 0.3)
- 안정적 상관 → bootstrap CI 폭이 좁은 쌍 (CI_width < 0.1)

**클러스터 기반 인사이트:**
- 최적 k 추천 근거 (실루엣 스코어 + 엘보우)
- 군집별 핵심 차별화 특성 상위 3개
- 소수 군집(< 전체의 5%) → 이상치 군집 의심, 별도 분석 권고
- 군집 간 크기 편차 → 불균형 정도와 대응 전략

**이상치 기반 인사이트:**
- 다변량 합의(consensus ≥ 2/3) 이상치 비율 및 특성
- 단변량 vs 다변량 이상치 불일치 → 변수 간 상호작용 이상치
- 이상치 제거 전후 통계량 변화 추정

**결측 기반 인사이트:**
- 결측 패턴(MCAR/MAR/MNAR) 진단 + 대체 전략 추천
- 컬럼 간 결측 상관(함께 결측인 컬럼 쌍) → 체계적 결측 의심
- 결측률 구간별 카운트 (0%, 0-5%, 5-20%, 20-50%, 50%+)

#### 2.1.3 인사이트 우선순위화

```python
@dataclass
class Insight:
    type: InsightType          # FINDING | WARNING | RECOMMENDATION | OPPORTUNITY
    severity: str              # critical | high | medium | low
    category: str              # distribution | correlation | cluster | anomaly | missing | quality | feature
    title: str                 # 한줄 제목
    description: str           # 상세 설명
    affected_columns: list[str]  # 관련 컬럼
    evidence: dict[str, Any]   # 근거 데이터
    action_items: list[str]    # 구체적 조치 항목
    priority_score: float      # 0~1 산출 점수 (정렬용)

class InsightEngine:
    """모든 분석 결과를 종합하여 인사이트를 생성·우선순위화한다."""

    def generate(self, stats: StatsResult, schema: DataSchema) -> list[Insight]:
        """모든 규칙 엔진을 실행하고 인사이트를 우선순위 역순으로 정렬하여 반환."""

    def _score_priority(self, insight: Insight) -> float:
        """심각도 × 영향범위(affected_columns 수) × 실행가능성 가중으로 점수 산정."""
```

#### 2.1.4 HTML 렌더링

```
┌──────────────────────────────────────────────────────────────┐
│  📊 Key Insights                          [Show All / Top 10] │
├──────────────────────────────────────────────────────────────┤
│  🔴 CRITICAL (2)                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ ⚠️ 3 column pairs show multicollinearity (VIF>10)       ││
│  │    → col_a ↔ col_b (r=0.97), col_c ↔ col_d (r=0.95)   ││
│  │    💡 Action: Consider removing one from each pair       ││
│  └──────────────────────────────────────────────────────────┘│
│  🟡 HIGH (5)                                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ 🔍 Column 'price' follows lognormal distribution         ││
│  │    → skew=2.3, best-fit: lognorm (AIC=-1234)            ││
│  │    💡 Action: Apply log transform before modeling        ││
│  └──────────────────────────────────────────────────────────┘│
│  🟢 MEDIUM (8)  │  ⚪ LOW (12)                               │
└──────────────────────────────────────────────────────────────┘
```

**인사이트 패널 위치**: 각 subset의 Basic 탭 Overview 섹션 바로 아래, 모든 분석 섹션보다 앞에 위치.

---

### Phase 2: 교차 분석 (Cross-Dimensional Analysis) — `stats/cross_analysis.py`

> **목적**: 개별 분석을 넘어 분석 차원 간 교차점에서 발현하는 복합 패턴을 탐지한다.

#### 2.2.1 교차 분석 유형

| # | 교차 축 | 기법 | 근거 | 산출물 |
|---|---------|------|------|--------|
| **X1** | 이상치 × 클러스터 | 클러스터별 이상치 분포 불균형 분석 | 특정 군집에 이상치 집중 → 해당 군집이 에러 데이터일 수 있음 | cluster_id별 anomaly_rate 테이블 + 막대 차트 |
| **X2** | 결측 × 상관 | 결측 여부를 더미(0/1)로 변환 후 다른 컬럼과의 상관 | 결측이 무작위(MCAR)가 아닌 체계적(MAR/MNAR) 여부 판단 | missing_indicator ↔ features 상관행렬 |
| **X3** | 분포 × 이상치 | 분포 꼬리 형태별 이상치 탐지 방법 적합성 | Heavy-tail 분포에서 IQR 방법은 과탐지 → Mahalanobis/IF 우선 권고 | 컬럼별 권장 이상치 탐지 방법 테이블 |
| **X4** | 클러스터 × 상관 | 군집별 상관 구조 비교 (within-cluster correlation) | Simpson's paradox 탐지: 전체-수준과 군집-수준 상관이 역전되는 경우 | 군집별 상관행렬 + 전체 상관 대비 차이 히트맵 |
| **X5** | 특성중요도 × 결측 | 중요 특성에서의 결측률 교차 확인 | 가장 중요한 컬럼에 결측이 많으면 심각한 정보 손실 | 중요도 vs 결측률 scatter + 경고 |
| **X6** | 차원축소 × 클러스터 | t-SNE/UMAP 임베딩 공간에서 클러스터 레이블 오버레이 | 클러스터 분리도의 시각적 확인, 클러스터 경계 명확성 | 2D scatter (색상=클러스터, 마커=이상치) |

#### 2.2.2 Simpson's Paradox 탐지기 (X4 상세)

```python
class SimpsonParadoxDetector:
    """
    전체 데이터에서의 상관 방향과 군집 내 상관 방향이 반전되는
    Simpson's Paradox를 자동 탐지한다.

    근거: Simpson(1951), Blyth(1972)
    """

    def detect(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        numeric_cols: list[str],
    ) -> pd.DataFrame:
        """
        Returns DataFrame:
            col_a, col_b, overall_corr, cluster_corrs (dict),
            is_paradox (bool), paradox_strength (float)
        """
```

#### 2.2.3 결측-상관 분석기 (X2 상세)

```python
class MissingCorrelationAnalyzer:
    """
    각 컬럼의 결측 여부를 이진 지시자(indicator)로 변환한 뒤,
    원본 수치 컬럼들과의 상관을 계산하여 결측의 체계성을 진단한다.

    높은 상관 → MAR (Missing At Random): 다른 변수 값에 의존하는 결측
    낮은 상관 → MCAR (Missing Completely At Random): 완전 무작위 결측

    근거: Little & Rubin (2002), *Statistical Analysis with Missing Data*
    """

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Returns:
            missing_indicator_corr: DataFrame — 결측 지시자 × 수치 컬럼 상관 행렬
            mar_suspects: list[dict] — |corr| > 0.3인 (결측컬럼, 상관컬럼, 상관계수) 쌍
            mcar_test_result: dict — Little's MCAR test 결과 (가능한 경우)
            imputation_strategy: dict — 컬럼별 추천 대체 전략
        """
```

---

### Phase 3: 컬럼 역할 자동 분류 — `stats/column_role.py`

> **목적**: 각 컬럼이 데이터셋 내에서 어떤 역할(ID, 타겟, 피처, 시간, 텍스트 등)을 하는지 자동 추론한다.

#### 2.3.1 역할 분류 체계

| 역할 | 탐지 기준 | 의미 | 후속 조치 |
|------|-----------|------|-----------|
| `ID` | 유니크율 > 95%, 이름에 id/key/index 포함, 단조증가 패턴 | 개체 식별자 → ML에서 제거 필요 | "제거 권장" 인사이트 |
| `TIMESTAMP` | datetime 타입 or 단조증가 정수 + 이름에 time/date/ts 포함 | 시간 인덱스 → 시계열 분석 가능 | "시계열 분석 활성화" 인사이트 |
| `CATEGORICAL_FEATURE` | 기존 categorical 타입 + 유니크율 < 5% | 범주형 입력 변수 | 인코딩 전략 추천 |
| `ORDINAL_FEATURE` | 정수형 + 유니크 값이 연속적 + 이름에 level/grade/rating 포함 | 순서형 입력 변수 | 레이블 인코딩 추천 |
| `NUMERIC_FEATURE` | 기존 numeric 타입 + ID/TIMESTAMP 아닌 것 | 연속형 입력 변수 | 스케일링/정규화 추천 |
| `TEXT` | 기존 text 타입 | 자유 텍스트 → NLP 파이프라인 필요 | 텍스트 최소 프로파일(평균 길이, 어휘 크기) |
| `BINARY` | 유니크 값이 정확히 2개 | 이진 변수 | 클래스 균형 확인 |
| `CONSTANT` | 유니크 값이 1개 | 정보 없음 → 제거 필요 | "제거 필수" 인사이트 |
| `TARGET_CANDIDATE` | 이름에 target/label/y/class 포함 + 낮은 cardinality | 잠재적 타겟 변수 후보 | ML 문맥에서 활용 |

#### 2.3.2 구현

```python
@dataclass
class ColumnRole:
    column: str
    primary_role: str              # 위 역할 중 하나
    confidence: float              # 0~1 확신도
    secondary_role: str | None     # 보조 역할 (예: ORDINAL이면서 TARGET_CANDIDATE)
    properties: dict[str, Any]     # 역할 판단 근거 증빙

class ColumnRoleClassifier:
    """모든 컬럼의 역할을 자동 추론한다."""

    def classify(self, df: pd.DataFrame, schema: DataSchema) -> list[ColumnRole]:
        """각 컬럼에 대해 역할을 판정하고 확신도를 산출한다."""

    def summary(self) -> pd.DataFrame:
        """컬럼 × 역할 요약 테이블을 반환한다."""
```

---

### Phase 4: ML 준비도 평가 — `stats/ml_readiness.py`

> **목적**: 데이터셋이 ML 파이프라인에 투입되기 전에 얼마나 준비되어 있는지를 다차원으로 평가한다.

#### 2.4.1 평가 차원

| 차원 | 세부 지표 | 가중치 | 근거 |
|------|-----------|--------|------|
| **완전성 (Completeness)** | 전체 결측률, 고결측 컬럼 비율, 행 단위 결측 분포 | 25% | 결측이 많으면 대체/제거 전처리 필수 |
| **일관성 (Consistency)** | 타입 혼합 컬럼 수, 이상 범위 값 비율, id-like 컬럼 비율 | 15% | 타입 불일치/이상값은 모델 오류 유발 |
| **균형성 (Balance)** | 범주 불균형(Gini), 이상치 비율, 클래스 비율(타겟 존재 시) | 15% | 불균형 데이터는 편향된 학습 유발 |
| **정보성 (Informativeness)** | 상수 컬럼 비율, 중복 행 비율, 평균 MI, 분산 분포 | 20% | 정보 없는 피처는 노이즈만 추가 |
| **독립성 (Independence)** | 다중공선성(VIF>10 비율), 완전 상관 쌍 수, 평균 상관 | 15% | 높은 공선성은 모델 불안정 유발 |
| **규모성 (Scale)** | 행 수 대비 컬럼 수(차원의 저주), 유효 피처 수 vs 표본 수 | 10% | n << p 상황은 과적합 위험 |

#### 2.4.2 종합 점수 및 등급

```
ML Readiness Score: 78.5 / 100
Grade: B+ (Good — minor preprocessing needed)

┌─────────────────────────────────────────┐
│  Completeness     ████████████░░  85%   │
│  Consistency      ██████████████  95%   │
│  Balance          ██████████░░░░  72%   │
│  Informativeness  ███████████░░░  80%   │
│  Independence     ██████░░░░░░░░  50%   │ ← 주의 필요
│  Scale            █████████████░  90%   │
└─────────────────────────────────────────┘

Blocking Issues (must fix before ML):
 ⛔ 2 columns have >50% missing — drop or impute
 ⛔ VIF > 100 detected for 'col_x' — remove or combine

Improvement Suggestions:
 💡 Apply log transform to 3 skewed features
 💡 Consider SMOTE for class imbalance (minority: 8%)
 💡 Remove 2 constant columns
```

#### 2.4.3 구현

```python
@dataclass
class ReadinessScore:
    overall: float               # 0~100
    grade: str                   # A+, A, B+, B, C+, C, D, F
    dimensions: dict[str, float] # 각 차원별 0~100
    blocking_issues: list[str]   # 반드시 해결해야 할 문제
    suggestions: list[str]       # 권장 개선 사항
    details: dict[str, Any]      # 상세 근거 데이터

class MLReadinessEvaluator:
    """데이터셋의 ML 준비도를 다차원으로 평가한다."""

    def evaluate(
        self, df, schema, stats: StatsResult
    ) -> ReadinessScore:
        """이전 분석 결과를 활용하여 ML 준비도를 산정한다."""
```

---

### Phase 5: 기존 모듈 개선

#### 2.5.1 statistical_tests.py 개선

| 개선 항목 | 현재 | 변경 | 근거 |
|-----------|------|------|------|
| **Kruskal-Wallis 시맨틱** | 수치 컬럼 간 비교 (의미 없음) | 범주형 변수를 그룹 변수로 활용하여 수치 컬럼의 그룹 간 차이 검정 | 올바른 통계적 사용법 |
| **다중검정 보정** | 없음 | Benjamini-Hochberg FDR 보정 (모든 쌍별 검정에 적용) | Benjamini & Hochberg(1995) |
| **효과 크기** | 없음 | Cohen's d (연속), Cramér's V (범주), η² (ANOVA/KW), rank-biserial r (MW) | 실무적 유의미성 판단에 필수 |
| **Bonferroni-adjusted significance** | 없음 | adjusted_p 컬럼 추가 + significance star 업데이트 | 대량 다중비교 시 Type I 오류 통제 |

#### 2.5.2 quality.py 개선

| 개선 항목 | 현재 | 변경 |
|-----------|------|------|
| **consistency()** | `series.apply(type).nunique()` — O(n) 느림 | `series.dtype` 기반 빠른 검사 + 도메인 규칙 (ex: 음수 나이, 미래 날짜) |
| **새 차원: Timeliness** | 없음 | datetime 컬럼의 최신성(recency) + 시간 범위 적절성 평가 |
| **새 차원: Conformity** | 없음 | 값 범위, 패턴 일치(regex), 도메인 규격 준수율 평가 |
| **컬럼 품질 세분화** | 4차원 집계 | 각 차원별 컬럼 레벨 점수 표시 |

#### 2.5.3 Data Profile 섹션 강화 (`report/generator.py`)

현재 7개 숫자 카드만 표시하는 Data Profile을 다음으로 확장:

| 구성요소 | 내용 |
|----------|------|
| **Dataset Overview Cards** | 행 수, 컬럼 수, 메모리, 결측률, 중복률, 수치/범주 비율 |
| **Column Role Table** | 각 컬럼의 역할(ID/Feature/Target/Time), 타입, 유니크, 결측률, 샘플값 3개 |
| **ML Readiness Dashboard** | 6차원 레이더 차트 + 종합 점수/등급 + blocking issues + suggestions |
| **Health Radar Chart** | 방사형 차트: completeness, consistency, outlier_ratio(반전), skewness_balance, correlation_health, duplicate_freedom |
| **Type Distribution Donut** | 컬럼 타입 비율 도넛 차트 (numeric/categorical/text/datetime/boolean) |
| **Memory Breakdown** | 컬럼별 메모리 사용량 수평 막대 (Top-10 heavy columns) |
| **Sample Data Preview** | 첫 5행 + 마지막 5행 (민감 데이터 마스킹 옵션) |

#### 2.5.4 누락된 차트 등록 (report/generator.py)

| 섹션 | 누락 차트 | 등록 방법 |
|------|-----------|-----------|
| **Dim. Reduction** | t-SNE scatter, UMAP scatter, Factor loadings heatmap, Feature contribution bar | `chart_keys` 맵에 추가 + VizResult에 해당 plot 메서드 구현 |
| **Feature Insights** | Interaction strength bar, Monotonic gap scatter, Binning comparison, Cardinality distribution | `chart_keys` 맵에 추가 + `viz/insight_plots.py` 구현 |
| **Statistical Tests** | Group comparison boxplots, p-value summary bar, Effect size forest plot | `chart_keys` 맵에 추가 + 기존 plotter 확장 |
| **Cross Analysis** | 이상치×클러스터 bar, 결측×상관 heatmap, Simpson paradox highlight, 중요도×결측 scatter, 통합 2D scatter | `viz/cross_plots.py`에서 구현 |

---

### Phase 6: 시각화 확장

#### 2.6.1 신규 시각화 모듈

**`viz/insight_plots.py`:**
| 차트 | 용도 | 기법 |
|------|------|------|
| `insight_severity_bar()` | 인사이트 심각도별 개수 bar chart | 수평 막대, 색상 코딩 |
| `interaction_strength_bar()` | 상호작용 세기 상위 N개 | 수평 막대 |
| `monotonic_gap_scatter()` | Pearson vs Spearman 차이 scatter | X=Pearson, Y=Spearman, 대각선 기준 |
| `binning_comparison()` | Equal-width vs equal-freq 엔트로피 비교 | 병렬 막대 |
| `cardinality_distribution()` | 컬럼별 cardinality 분포 | 히스토그램 + 인코딩 전략 색상 |

**`viz/cross_plots.py`:**
| 차트 | 용도 |
|------|------|
| `anomaly_by_cluster_bar()` | 군집별 이상치 비율 막대 |
| `missing_correlation_heatmap()` | 결측 지시자 × 수치 컬럼 상관 히트맵 |
| `simpson_paradox_highlight()` | 전체 vs 군집별 상관 비교 scatter (방향 반전 강조) |
| `importance_vs_missing_scatter()` | X=중요도, Y=결측률, 크기=유니크 수 |
| `unified_2d_scatter()` | t-SNE/UMAP 2D에 클러스터 색상 + 이상치 마커 통합 |

**`viz/dimreduction_plots.py`:**
| 차트 | 용도 |
|------|------|
| `tsne_scatter()` | t-SNE 2D scatter (클러스터 라벨 오버레이 지원) |
| `umap_scatter()` | UMAP 2D scatter |
| `factor_loadings_heatmap()` | Factor Analysis loadings 히트맵 |
| `feature_contribution_bar()` | PCA 기반 feature 기여도 bar chart |

#### 2.6.2 Health Radar Chart

```python
def health_radar_chart(quality_scores: dict, ml_readiness: ReadinessScore) -> plt.Figure:
    """
    6축 방사형 차트:
    - Completeness (결측 기반)
    - Consistency (타입 일관성)
    - Outlier Freedom (1 - 이상치 비율)
    - Distribution Health (정규성/대칭성)
    - Correlation Health (다중공선성 없음 정도)
    - Duplicate Freedom (1 - 중복률)

    중앙에 종합 점수 표시, 각 축에 0~100 스케일.
    """
```

---

## 3. 교차분석 섹션의 HTML 배치

Advanced 서브탭에 2개 신규 탭 추가:

```
[Basic] [Distribution+] [Correlation+] [Clustering] [Dim. Reduction]
[Feature Insights] [Anomaly+] [Stat Tests] [Cross Analysis ★] [Data Profile ★]
```

**Cross Analysis** 탭은 Phase 2의 6개 교차 분석을 포함:
```
Cross Analysis
├── Outlier × Cluster Distribution
├── Missing × Correlation (MAR Detection)
├── Distribution × Outlier Method Fitness
├── Cluster × Correlation (Simpson's Paradox Check)
├── Feature Importance × Missing Rate
└── Unified 2D Embedding (t-SNE/UMAP + Cluster + Anomaly overlay)
```

**Data Profile** 탭은 기존 7카드 → 풍부한 대시보드:
```
Data Profile ★ (Enhanced)
├── Overview Cards (확장)
├── Column Roles Table (신규)
├── ML Readiness Dashboard (신규)
│   ├── Radar Chart (6 dimensions)
│   ├── Score & Grade
│   ├── Blocking Issues
│   └── Suggestions
├── Health Radar Chart (신규)
├── Type Distribution Donut (신규)
├── Memory Breakdown Chart (신규)
└── Sample Data Preview (신규)
```

---

## 4. 인사이트 패널의 HTML 배치

**위치**: Basic 탭의 Overview 바로 아래 (모든 분석 섹션보다 앞)

```
Basic Tab
├── Overview
├── ★ Key Insights Panel ★  ← 신규 위치
│   ├── Executive Summary (1~2문장 총평)
│   ├── Critical Issues (접힘가능)
│   ├── Key Findings (접힘가능)
│   ├── Recommendations (접힘가능)
│   └── Opportunities (접힘가능)
├── Data Quality
├── Preprocessing
├── Descriptive Statistics
│   ...
```

---

## 5. 구현 순서 및 의존성 그래프

```
Phase 1: Insight Engine ─────────────────────────┐
  stats/insight_engine.py                         │
  (depends on: 기존 모든 stats 결과)               │
                                                  │
Phase 2: Cross Analysis ───────────────────────┐  │
  stats/cross_analysis.py                       │  │
  (depends on: clustering, anomaly, correlation,│  │
   missing, feature_importance, dimreduction)    │  │
                                                │  │
Phase 3: Column Role ──────────────────────────┐│  │
  stats/column_role.py                          ││  │
  (depends on: schema, descriptive)             ││  │
                                                ││  │
Phase 4: ML Readiness ─────────────────────────┤│  │
  stats/ml_readiness.py                        ││  │
  (depends on: quality, column_role, stats)    ││  │
                                               ││  │
Phase 5: 기존 모듈 개선 ──────────────────────── ││  │
  stats/statistical_tests.py (패치)             ││  │
  stats/quality.py (패치)                       ││  │
                                               ││  │
Phase 6: 시각화 확장 ── ─────────────────────── ┤│  │
  viz/insight_plots.py                         ││  │
  viz/cross_plots.py                           ││  │
  viz/dimreduction_plots.py                    ││  │
                                               ↓↓  ↓
Phase 7: 통합 ──────────────────────────────────────┘
  core/config.py (토글 추가)
  core/analyzer.py (새 모듈 호출 + VizResult 확장)
  report/generator.py (인사이트 패널, 교차분석 섹션, Data Profile 강화, 누락 차트)
  report/i18n.py (번역 키 추가)
```

**병렬 가능**: Phase 1·2·3은 서로 독립적으로 구현 가능
**순차 필수**: Phase 4(ML Readiness)는 Phase 3(Column Role) 완료 후 시작
**최종 통합**: Phase 7은 모든 모듈 완성 후

---

## 6. 기술 의존성

### 6.1 신규 패키지 필요 여부

| 필요 기능 | 패키지 | 필수여부 | 비고 |
|-----------|--------|----------|------|
| FDR 보정 | `scipy.stats` (이미 있음) | 이미 설치 | `scipy.stats.false_discovery_control` 또는 직접 Benjamini-Hochberg 구현 |
| 효과 크기 계산 | 직접 구현 | 새 코드 | Cohen's d, η², rank-biserial r — 수식이 단순하므로 외부 패키지 불필요 |
| Radar chart | `matplotlib` (이미 있음) | 이미 설치 | polar projection subplot으로 구현 |
| Donut chart | `matplotlib` (이미 있음) | 이미 설치 | `pie()` with `wedgeprops` |
| 교차분석 | `numpy`, `pandas`, `scipy` (이미 있음) | 이미 설치 | 기존 의존성만으로 충분 |

**결론: 새로운 외부 패키지 추가 불필요.** 기존 scipy + numpy + pandas + matplotlib + sklearn으로 100% 구현 가능.

### 6.2 성능 고려사항

| 신규 분석 | 복잡도 | 대응 전략 |
|-----------|--------|-----------|
| Insight Engine | O(1) — 이미 계산된 결과에서 규칙 적용 | 규칙 평가는 마이크로초 단위, 성능 무관 |
| Cross Analysis: 결측×상관 | O(n·d) | 기존 결측 분석 + 상관 결과 재활용 |
| Cross Analysis: Simpson's Paradox | O(k·d²) per cluster | k=max(10), d=max(15) → 무시 가능 |
| Column Role Classification | O(d) | 컬럼 수만큼, 룰 기반이므로 즉시 |
| ML Readiness | O(1) | 이미 계산된 통계량 조합 |
| Health Radar | O(1) | 단일 차트 렌더링 |
| 누락 차트 등록 | 기존과 동일 | 이미 VizResult에 존재하는 것을 등록만 |

**총 추가 분석 시간 예측**: 기존 Advanced 분석 대비 **+5~10%** 이내 (대부분 기존 결과 재활용)

---

## 7. 효과성 평가

### 7.1 Before vs After

| 카테고리 | Before (현재) | After (Enhancement) | 변화 |
|----------|--------------|-------------------|----- |
| 자동 인사이트 | 0 | 4타입 × ~40개 규칙 → 5~30개 인사이트/데이터셋 | **신규** |
| 교차 분석 | 0 | 6종 교차 분석 | **신규** |
| 컬럼 역할 분류 | 0 (타입만 추론) | 9종 역할 자동 분류 | **신규** |
| ML 준비도 | 0 | 6차원 평가 + 등급 + blocking issues | **신규** |
| 통계 검정 엄밀성 | p-value만 | + 효과 크기 + 다중검정 보정 | **+200%** |
| Data Profile | 7 카드 | 종합 대시보드 (7섹션) | **+700%** |
| 시각화 | 30+ 차트 | +15종 추가 | **+50%** |
| 보고서 섹션 | 21 섹션 | +2 탭 (Cross Analysis, Data Profile 강화) | **+10%** |

### 7.2 사용자 경험 향상

| 사용자 질문 | Before | After |
|------------|--------|-------|
| "이 데이터에서 뭐가 중요해?" | 직접 표/차트 해석 | 자동 인사이트 → 핵심 발견 즉시 파악 |
| "ML에 바로 쓸 수 있어?" | 품질 점수만 참고 | ML Readiness 등급 + blocking issues 목록 |
| "이상치가 특정 그룹에 집중되나?" | 개별 이상치/클러스터 결과 따로 확인 | 교차분석 → 군집별 이상치 분포 한눈에 |
| "결측이 무작위인가?" | 결측률만 표시 | 결측×상관 분석 → MAR/MCAR 자동 진단 |
| "어떤 전처리가 필요해?" | 경고 목록 참고 | 인사이트 엔진 → 구체적 조치 항목 목록 |
| "Simpson's paradox는 없나?" | 확인 불가 | 교차분석 → 자동 탐지 + 시각화 |

### 7.3 분석 깊이 비교 (표준 EDA 도구 대비)

| 기능 | pandas-profiling | sweetviz | f2a v1 (현재) | f2a v2 (Enhancement) |
|------|-----------------|----------|--------------|---------------------|
| 기술 통계 | ★★★★ | ★★★ | ★★★★★ | ★★★★★ |
| 분포 분석 | ★★★ | ★★★ | ★★★★★ | ★★★★★ |
| 상관 분석 | ★★★ | ★★★★ | ★★★★★ | ★★★★★ |
| 이상치 탐지 | ★★ | ★★ | ★★★★★ | ★★★★★ |
| 클러스터링 | ✗ | ✗ | ★★★★ | ★★★★ |
| 차원축소 | ✗ | ✗ | ★★★★ | ★★★★★ |
| 교차분석 | ✗ | ✗ | ✗ | ★★★★ |
| 자동 인사이트 | ★★ | ★★★ | ✗ | ★★★★★ |
| ML 준비도 | ✗ | ✗ | ✗ | ★★★★ |
| 컬럼 역할 분류 | ✗ | ✗ | ✗ | ★★★★ |
| 통계 검정 엄밀성 | ★★ | ★ | ★★★ | ★★★★★ |
| i18n | ✗ | ✗ | ★★★★★ | ★★★★★ |
| HuggingFace 지원 | ✗ | ✗ | ★★★★★ | ★★★★★ |

---

## 8. 파일별 구현 명세 (Summary)

| # | 파일 | 동작 | 예상 LOC | 의존성 |
|---|------|------|---------|--------|
| 1 | `stats/insight_engine.py` | 신규 | ~500 | StatsResult, DataSchema |
| 2 | `stats/cross_analysis.py` | 신규 | ~450 | numpy, pandas, scipy |
| 3 | `stats/column_role.py` | 신규 | ~250 | schema, descriptive stats |
| 4 | `stats/ml_readiness.py` | 신규 | ~350 | quality, column_role, StatsResult |
| 5 | `stats/statistical_tests.py` | 수정 | +~120 | scipy.stats |
| 6 | `stats/quality.py` | 수정 | +~80 | pandas |
| 7 | `viz/insight_plots.py` | 신규 | ~300 | matplotlib |
| 8 | `viz/cross_plots.py` | 신규 | ~350 | matplotlib, numpy |
| 9 | `viz/dimreduction_plots.py` | 신규 | ~200 | matplotlib |
| 10 | `core/config.py` | 수정 | +~30 | — |
| 11 | `core/analyzer.py` | 수정 | +~200 | 새 모듈 import |
| 12 | `report/generator.py` | 수정 | +~400 | HTML/CSS/JS |
| 13 | `report/i18n.py` | 수정 | +~200 | — |
| **합계** |  |  | **~3,430 LOC** |  |

---

## 9. 검증 계획

| 검증 항목 | 방법 | 기준 |
|-----------|------|------|
| 인사이트 품질 | 합성 데이터(정규/비정규/결측/이상치) + lerobot/roboturk 실데이터 | 알려진 패턴이 인사이트로 탐지되는지 |
| 교차분석 정확성 | Simpson's paradox 합성 데이터 | 반전 탐지 100% |
| ML 준비도 유효성 | 품질 좋은/나쁜 데이터셋 대비 | 점수 차이가 직관과 일치 |
| 통계 검정 보정 | 시뮬레이션(H0 하 1000회) | FDR ≤ 0.05 |
| 기존 테스트 통과 | `pytest git_action/tests/` | 전체 통과 |
| HTML 렌더링 | 생성된 리포트 브라우저 확인 | 모든 섹션 올바르게 표시 |
| 성능 | 10만행 × 50컬럼 | 전체 분석 < 120초 |

---

*이 계획은 f2a를 단순 통계 리포트 생성기에서 **인텔리전트 데이터 분석 엔진**으로 진화시키기 위한 로드맵이다.*
*모든 추가 기능은 학술적 근거 위에 실무적 가치를 제공하며, 기존 의존성만으로 구현 가능하다.*
