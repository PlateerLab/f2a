# f2a Advanced Analysis Plan

> **목적**: ML 논문/기법 기반의 고급 분석 기능을 체계적으로 설계하고, HTML 리포트에 2-depth 탭 구조로 제공한다.

---

## 1. 현재 상태 분석 (As-Is)

### 1.1 현재 구현된 분석 (Basic Report)

| 영역 | 기법 | 비고 |
|------|------|------|
| **Descriptive** | count, missing, unique, mean, median, std, SE, CV, MAD, min/max/range, p5/q1/q3/p95, IQR, skewness, kurtosis | 16개 수치 지표 |
| **Distribution** | Shapiro-Wilk, D'Agostino, KS, Anderson-Darling, skew/kurt 분류 | 4개 정규성 검정 |
| **Correlation** | Pearson, Spearman, Kendall, Cramér's V, VIF | 5종 상관 분석 |
| **Missing** | column summary, row distribution, total ratio | 기초 결측 분석 |
| **Outlier** | IQR method, Z-score method | 2종 이상치 탐지 |
| **Categorical** | frequency, entropy, chi-square independence | 범주 분석 |
| **Feature Importance** | variance ranking, mean abs correlation, mutual information | 3종 중요도 |
| **PCA** | StandardScaler + PCA, scree, loadings | 기초 차원축소 |
| **Duplicates** | exact duplicates, column uniqueness | 중복 탐지 |
| **Quality** | completeness, uniqueness, consistency, validity (weighted) | 4차원 품질 |

### 1.2 현재 HTML 구조

```
Header
├── [1-depth tabs: subset/split 선택]  ← 현재 유일한 탭 계층
│   ├── Overview
│   ├── Data Quality
│   ├── Preprocessing
│   ├── Descriptive Statistics
│   ├── Distribution Analysis
│   ├── Correlation Analysis
│   ├── Missing Data
│   ├── Outlier Detection
│   ├── Categorical Analysis
│   ├── Feature Importance
│   ├── PCA
│   ├── Duplicates
│   └── Warnings
Footer
```

---

## 2. Advanced 분석 기법 설계 (To-Be)

### 2.1 새로운 HTML 2-Depth 탭 구조

```
Header
├── [1-depth: subset/split 선택]       ← 기존
│   ├── [2-depth: Basic | Advanced]    ← 신규
│   │   ├── Basic → 기존 모든 섹션 그대로
│   │   └── Advanced
│   │       ├── A1. Advanced Distribution
│   │       ├── A2. Advanced Correlation
│   │       ├── A3. Clustering Analysis
│   │       ├── A4. Dimensionality Reduction
│   │       ├── A5. Feature Engineering Insights
│   │       ├── A6. Anomaly Detection
│   │       ├── A7. Statistical Tests
│   │       └── A8. Data Profiling Summary
Footer
```

---

### 2.2 Advanced 탭 상세 설계

---

#### A1. Advanced Distribution Analysis

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **Kernel Density Estimation (KDE) bandwidth selection** | Silverman(1986), Scott's rule | `scipy.stats.gaussian_kde`로 최적 bandwidth 자동 추정, KDE curve + histogram overlay | 데이터의 실제 분포 형태를 비모수적으로 파악 |
| **Best-fit distribution matching** | D'Agostino & Stephens(1986) | `scipy.stats`의 주요 분포(norm, lognorm, exponential, gamma, beta, weibull, uniform) 피팅 후 AIC/BIC 비교 | 각 컬럼이 어떤 이론적 분포에 가장 가까운지 자동 식별 |
| **Jarque-Bera test** | Jarque & Bera(1987) | `scipy.stats.jarque_bera` — skewness+kurtosis 기반 정규성 검정 | 기존 4개 검정에 추가, 대표본에 특히 유효 |
| **Power transformation recommendation** | Box-Cox(1964), Yeo-Johnson(2000) | `scipy.stats.boxcox`/`yeojohnson`으로 변환 후 skewness 변화량 측정 | 어떤 변환이 정규성을 개선하는지 자동 추천 |
| **Empirical CDF** | Kolmogorov(1933) | `statsmodels.distributions.empirical_distribution.ECDF` 또는 직접 step plot | 데이터의 누적 분포를 직관적으로 시각화 |

**시각화:**
- KDE overlay histograms (bandwidth comparison)
- Best-fit distribution overlay plot (데이터 + 최적 분포 곡선)
- Power transformation before/after comparison
- ECDF step plots

---

#### A2. Advanced Correlation Analysis

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **Partial correlation** | Fisher(1924) | 다른 변수를 제어한 상태에서의 순수 상관. inverse correlation matrix에서 추출 | 교란 변수 제거한 진정한 관계 파악 |
| **Distance correlation** | Székely et al.(2007) | `dcor` 라이브러리 또는 직접 구현. 비선형 관계까지 감지 | Pearson이 놓치는 비선형 의존성 탐지 |
| **Mutual Information heatmap** | Shannon(1948), Kraskov et al.(2004) | sklearn `mutual_info_regression`으로 전체 컬럼 쌍 MI 행렬 생성 | 비선형 정보 공유량의 정량적 시각화 |
| **Correlation stability (bootstrap)** | Efron(1979) | 상관계수의 bootstrap 신뢰구간 (95% CI) 계산 | 상관 추정의 신뢰도/안정성 평가 |
| **Correlation network graph** | Graph theory | 상관 threshold 초과 쌍을 node-edge로 시각화 | 변수 간 관계 구조의 직관적 네트워크 파악 |

**시각화:**
- Partial correlation heatmap
- MI heatmap
- Bootstrap correlation CI forest plot
- Correlation network graph (matplotlib `networkx` layout)

---

#### A3. Clustering Analysis

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **K-Means + Elbow/Silhouette** | MacQueen(1967), Rousseeuw(1987) | sklearn `KMeans` (k=2~10) + inertia elbow + silhouette score → optimal k 자동 결정 | 데이터의 자연 군집 구조 탐색 |
| **DBSCAN** | Ester et al.(1996) | sklearn `DBSCAN` with automated eps (k-distance graph) | 밀도 기반 클러스터링, 노이즈/이상치 자연 분리 |
| **Hierarchical clustering (dendrogram)** | Ward(1963) | `scipy.cluster.hierarchy.linkage` + dendrogram | 계층적 구조 시각화, 적절한 컷 레벨 참고 |
| **Cluster profiling** | — | 군집별 평균/분포 요약 테이블 생성 | 각 군집의 특성 자동 프로파일링 |

**시각화:**
- Elbow plot + Silhouette score plot
- 2D PCA scatter with cluster labels (color-coded)
- DBSCAN result scatter (noise = gray)
- Dendrogram
- Cluster profile radar/bar chart

---

#### A4. Dimensionality Reduction (확장)

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **t-SNE** | van der Maaten & Hinton(2008) | sklearn `TSNE(n_components=2, perplexity=30)` | 고차원 데이터의 2D 비선형 임베딩으로 군집 시각화 |
| **UMAP** | McInnes et al.(2018) | `umap-learn` 라이브러리 (optional dependency) | t-SNE보다 빠르고 전역 구조 보존 |
| **Factor Analysis** | Spearman(1904), Thurstone(1935) | sklearn `FactorAnalysis` — 잠재 요인 추출 + loadings | PCA와 달리 잠재 변수 모델, 해석력 우수 |
| **Explained variance per feature** | Kaiser criterion(1960) | 각 원본 feature가 top-k PC에 기여하는 분산 비율 | feature-level 중요도의 차원축소 관점 제공 |

**시각화:**
- t-SNE 2D scatter (cluster labels overlay 가능)
- UMAP 2D scatter
- Factor loadings heatmap
- Feature contribution stacked bar chart

---

#### A5. Feature Engineering Insights

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **Interaction detection** | Friedman & Popescu(2008) | 수치 컬럼 쌍의 곱/비율 생성 후 분산/상관 분석 | 유망한 interaction feature 자동 발견 |
| **Monotonic relationship detection** | Spearman rho | Spearman vs. Pearson 차이로 비선형 단조성 판별 | 변환이 필요한 비선형 관계 식별 |
| **Binning analysis** | Dougherty et al.(1995) | 수치 컬럼의 equal-width/equal-freq 빈 생성, 빈별 엔트로피 비교 | 이산화 전략 선택 도움 |
| **Cardinality analysis** | — | 범주형 컬럼의 유니크 비율별 인코딩 전략 추천 (one-hot / target / ordinal) | 전처리 파이프라인 설계 자동 가이드 |
| **Target leakage detection** | Kaufman et al.(2012) | 수치 컬럼 중 다른 컬럼과 r>0.99 또는 MI≈max인 쌍 경고 | 데이터 누수 조기 발견 |

**시각화:**
- Top-N interaction feature 분포 히스토그램
- Spearman vs Pearson 차이 bar chart
- Encoding strategy recommendation 테이블

---

#### A6. Anomaly Detection (확장)

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **Isolation Forest** | Liu et al.(2008) | sklearn `IsolationForest` → anomaly score per row | 다변량 이상치 탐지 (IQR/Z-score는 단변량) |
| **Local Outlier Factor (LOF)** | Breunig et al.(2000) | sklearn `LocalOutlierFactor` → LOF score per row | 밀도 기반 국소 이상치, 군집 밖의 점 탐지 |
| **Mahalanobis distance** | Mahalanobis(1936) | 공분산 기반 다변량 거리, chi-squared 임계값 | 상관 구조를 고려한 다변량 이상치 |
| **Anomaly summary** | — | 다수 방법의 consensus (≥2 방법에서 anomaly → 고확률) | 단일 방법 의존 제거, 견고한 이상치 판정 |

**시각화:**
- Isolation Forest anomaly score 분포 히스토그램
- LOF score scatter (2D PCA 공간에서)
- Mahalanobis distance 히스토그램 with chi-squared 임계선
- Consensus anomaly heatmap (row × method)

---

#### A7. Statistical Tests

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **Levene's test (등분산)** | Levene(1960) | 범주별로 수치 컬럼의 분산 동질성 검정 | ANOVA 전제조건 확인 |
| **Kruskal-Wallis test** | Kruskal & Wallis(1952) | 비모수 다집단 중위수 비교 | 비정규 분포에서의 집단 차이 검정 |
| **Mann-Whitney U test** | Mann & Whitney(1947) | 이진 범주와 수치 컬럼 간 비모수 검정 | 두 집단 차이의 비모수 평가 |
| **Chi-square goodness of fit** | Pearson(1900) | 범주형 컬럼의 균등 분포 검정 | 범주 분포의 편향 정도 정량 평가 |
| **Grubbs' test** | Grubbs(1950) | 단일 이상치의 통계적 유의성 검정 | 극단값의 통계적 유의미성 판별 |
| **Stationarity (ADF test)** | Dickey & Fuller(1979) | 시계열 컬럼의 단위근 검정 (`statsmodels.tsa`) | 시계열 정상성 자동 판단 |

**시각화:**
- Group comparison boxplots (Kruskal-Wallis/Mann-Whitney와 함께)
- Test results summary table with p-values and significance stars
- Levene test bar chart per column

---

#### A8. Data Profiling Summary

| 기법 | 근거 | 구현 계획 | 효과 |
|------|------|-----------|------|
| **Automated insight generation** | AutoEDA literature | 모든 분석 결과를 종합하여 자연어 인사이트 생성 | 비전문가도 핵심 발견 사항을 즉시 파악 |
| **Feature type recommendation** | — | 각 컬럼의 분포/유니크/결측 패턴으로 최적 ML 타입 추천 | ML 파이프라인 설계 가이드 |
| **Dataset complexity scoring** | Ho & Basu(2002) | 차원수, 클래스 수, 불균형도, 상관 구조 → 복잡도 점수 | 데이터셋 난이도의 정량적 평가 |
| **Overall health dashboard** | — | 전체 분석 결과의 1-page 대시보드 (트래픽 라이트 시스템) | 데이터 상태의 즉각적 파악 |

**시각화:**
- Health score radar chart (6 축: completeness, consistency, outlier ratio, skewness, correlation, duplicates)
- Insight cards (자동 생성된 주요 발견 사항)
- Feature type recommendation table

---

## 3. 기술 의존성 분석

### 3.1 새로 필요한 패키지

| 패키지 | 용도 | 필수 여부 | 비고 |
|--------|------|-----------|------|
| `scikit-learn` | K-Means, DBSCAN, IsolationForest, LOF, FactorAnalysis, t-SNE, MI | **이미 설치됨** | core dependency |
| `networkx` | Correlation network graph | Optional | `try/except` 처리 |
| `umap-learn` | UMAP 차원축소 | Optional | `try/except` 처리 |
| `statsmodels` | ADF test, ECDF | Optional | `try/except` 처리 |

**원칙:** `scikit-learn`과 기존 종속성(`scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`)만으로 A1~A8의 80%+ 구현 가능. `networkx`, `umap-learn`, `statsmodels`는 optional — 없으면 해당 분석을 건너뛰고 "library not available" 메시지 표시.

### 3.2 성능 고려사항

| 기법 | 시간 복잡도 | 대응 전략 |
|------|-------------|-----------|
| t-SNE | O(n²) | n>5000이면 샘플링 후 수행 |
| UMAP | O(n·log(n)) | n>10000이면 샘플링 |
| Isolation Forest | O(n·t·log(n)) | max_samples=min(256, n) 기본 |
| MI 행렬 | O(n·d²) | d>30이면 top-30 컬럼만 |
| Bootstrap CI | O(B·n) | B=1000, n>5000이면 샘플링 |
| K-Means elbow | O(k·n·d·iter) | k=2~10, max_iter=100 |
| Best-fit distribution | O(n·d_count) | 7개 분포만 피팅 |

---

## 4. 구현 계획

### Phase 1: 인프라 (2-depth 탭 구조)

1. **`AnalysisConfig` 확장** — `advanced: bool = True` 플래그 + `AdvancedConfig` sub-dataclass 추가
2. **HTML generator 2-depth 탭** — 기존 subset 탭 내부에 "Basic / Advanced" 서브탭 도입
3. **`StatsResult` 확장** — `advanced_stats: dict[str, Any]` 필드 추가
4. **`VizResult` 확장** — advanced plot 메서드 추가

### Phase 2: Advanced Stats 모듈 (4개 파일)

5. **`stats/advanced_distribution.py`** — best_fit, kde_bandwidth, jarque_bera, power_transform, ecdf
6. **`stats/advanced_correlation.py`** — partial_corr, mi_matrix, bootstrap_ci, correlation_network_data
7. **`stats/clustering.py`** — kmeans_analysis, dbscan_analysis, hierarchical, cluster_profiles
8. **`stats/statistical_tests.py`** — levene, kruskal_wallis, mann_whitney, chi_sq_goodness, grubbs, adf_stationarity

### Phase 3: Advanced Stats 모듈 (3개 파일)

9. **`stats/advanced_anomaly.py`** — isolation_forest, lof, mahalanobis, consensus_anomaly
10. **`stats/advanced_dimreduction.py`** — tsne, umap, factor_analysis, feature_contribution
11. **`stats/feature_insights.py`** — interaction_detection, monotonic_detection, binning_analysis, cardinality_analysis, leakage_detection

### Phase 4: Advanced Viz 모듈 (4개 파일)

12. **`viz/advanced_dist_plots.py`** — best_fit_overlay, power_transform_comparison, ecdf_plot, kde_bandwidth_comparison
13. **`viz/advanced_corr_plots.py`** — partial_corr_heatmap, mi_heatmap, bootstrap_ci_plot, network_graph
14. **`viz/cluster_plots.py`** — elbow_plot, silhouette_plot, cluster_scatter, dendrogram, cluster_profiles_chart
15. **`viz/advanced_anomaly_plots.py`** — isolation_forest_hist, lof_scatter, mahalanobis_hist, consensus_heatmap

### Phase 5: 통합

16. **Analyzer `_compute_advanced_stats()` 추가** — 각 advanced 모듈 호출
17. **VizResult advanced plot 메서드 추가** — 각 advanced viz 호출
18. **HTML generator advanced 섹션 빌더** — 8개 advanced 섹션 + 서브탭
19. **Data Profiling Summary (A8)** — insights 자동 생성 로직

### Phase 6: 마무리

20. **pyproject.toml 업데이트** — optional deps 추가
21. **`_METRIC_TIPS` 확장** — advanced 지표 tooltip 추가
22. **End-to-end 테스트** — 실제 데이터셋으로 전체 리포트 생성 검증

---

## 5. 효과성 평가

### 5.1 분석 범위 확장

| 카테고리 | Basic (현재) | + Advanced | 커버리지 증가 |
|----------|-----------|------------|---------------|
| 정규성/분포 검정 | 4종 | +3종 (JB, 7-dist fitting, power transform) | +75% |
| 상관 분석 | 5종 | +4종 (partial, MI matrix, bootstrap CI, network) | +80% |
| 이상치 탐지 | 2종 (단변량) | +3종 (다변량: IF, LOF, Mahalanobis) | +150% |
| 차원축소 | PCA 1종 | +3종 (t-SNE, UMAP, Factor Analysis) | +300% |
| 군집 분석 | 0종 | +3종 (K-Means, DBSCAN, Hierarchical) | 신규 |
| 통계 검정 | 4종 (정규성) | +6종 (등분산, 비모수, 적합도, Grubbs, ADF) | +150% |
| Feature 공학 | 3종 (중요도) | +5종 (interaction, monotonic, binning, cardinality, leakage) | +167% |
| Data profiling | 품질 점수 | +3종 (insights, type recommendation, complexity) | +300% |

### 5.2 실무적 가치

1. **비선형 관계 탐지**: Pearson/Spearman만으로는 포착 불가능한 비선형 의존성을 MI, distance correlation 으로 발견
2. **다변량 이상치**: IQR/Z-score는 단변량 — Isolation Forest와 LOF로 변수 간 상호작용 고려한 이상치 탐지
3. **군집 구조 발견**: 데이터의 자연 그룹을 자동 탐색, ML 모델링 전 데이터 이해도 극대화
4. **최적 분포 식별**: 각 변수의 이론적 분포를 자동 피팅하여 변환/모델링 전략 결정
5. **통계적 유의성**: 시각적 차이를 넘어 통계 검정으로 엄밀한 판단 근거 제공
6. **Feature 공학 자동화**: interaction feature, 인코딩 전략, 데이터 누수를 자동 탐지

### 5.3 학술적 근거 (Key References)

| # | 논문/방법 | 연도 | 핵심 기여 |
|---|-----------|------|-----------|
| 1 | Silverman, *Density Estimation for Statistics and Data Analysis* | 1986 | KDE bandwidth selection |
| 2 | Jarque & Bera, *Efficient tests for normality* | 1987 | JB normality test |
| 3 | Box & Cox, *An analysis of transformations* | 1964 | Power transformation |
| 4 | Yeo & Johnson, *A new family of power transformations* | 2000 | 음수 허용 power transform |
| 5 | Székely et al., *Measuring and testing dependence by correlation of distances* | 2007 | Distance correlation |
| 6 | Shannon, *A mathematical theory of communication* | 1948 | Mutual information |
| 7 | Efron, *Bootstrap methods: another look at the jackknife* | 1979 | Bootstrap CI |
| 8 | MacQueen, *Some methods for classification* | 1967 | K-Means |
| 9 | Ester et al., *A density-based algorithm (DBSCAN)* | 1996 | DBSCAN |
| 10 | Rousseeuw, *Silhouettes: a graphical aid* | 1987 | Silhouette score |
| 11 | van der Maaten & Hinton, *Visualizing data using t-SNE* | 2008 | t-SNE |
| 12 | McInnes et al., *UMAP: Uniform manifold approximation* | 2018 | UMAP |
| 13 | Liu et al., *Isolation forest* | 2008 | Isolation Forest |
| 14 | Breunig et al., *LOF: identifying density-based local outliers* | 2000 | LOF |
| 15 | Mahalanobis, *On the generalized distance in statistics* | 1936 | Mahalanobis distance |
| 16 | Fisher, *The distribution of the partial correlation coefficient* | 1924 | Partial correlation |
| 17 | Levene, *Robust tests for equality of variances* | 1960 | Levene's test |
| 18 | Kruskal & Wallis, *Use of ranks in one-criterion variance analysis* | 1952 | KW test |
| 19 | Dickey & Fuller, *Distribution of the estimators* | 1979 | ADF stationarity test |
| 20 | Ho & Basu, *Complexity measures of supervised classification problems* | 2002 | Dataset complexity |

---

## 6. 2-Depth 탭 UI 설계

### 6.1 탭 구조

```html
<!-- 1st depth: subset/split (기존) -->
<div class="tab-bar">
  <button class="tab-btn active">default / train</button>
  <button class="tab-btn">default / test</button>
</div>

<!-- 각 1st-depth 탭 내부에 2nd depth -->
<div class="tab-content active" id="tab-0">
  <div class="sub-tab-bar">
    <button class="sub-tab-btn active" data-target="tab-0-basic">📊 Basic</button>
    <button class="sub-tab-btn" data-target="tab-0-advanced">🔬 Advanced</button>
  </div>
  <div class="sub-tab-content active" id="tab-0-basic">
    <!-- 기존 모든 섹션 -->
  </div>
  <div class="sub-tab-content" id="tab-0-advanced">
    <!-- Advanced 8개 섹션 -->
  </div>
</div>
```

### 6.2 Advanced 탭 내부 네비게이션

Advanced 탭 내에 섹션 앵커 점프 네비게이션:
```
[Distribution+] [Correlation+] [Clustering] [Dim. Reduction]
[Feature Eng.] [Anomaly] [Statistical Tests] [Profiling]
```

---

*Generated by f2a analysis planning system*
