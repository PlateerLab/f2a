"""f2a 빠른 시작 예제.

이 스크립트는 f2a의 주요 기능을 보여줍니다.

Usage:
    python examples/quickstart.py
"""

import numpy as np
import pandas as pd

# ── 1. 샘플 데이터 생성 ─────────────────────────────────
np.random.seed(42)
n = 300

df = pd.DataFrame(
    {
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(55000, 18000, n).round(2),
        "score": np.random.uniform(0, 100, n).round(1),
        "city": np.random.choice(["서울", "부산", "대구", "인천", "광주", "대전"], n),
        "grade": np.random.choice(["A", "B", "C", "D"], n, p=[0.2, 0.35, 0.3, 0.15]),
    }
)

# 약간의 결측치 추가
df.loc[np.random.random(n) < 0.08, "income"] = np.nan
df.loc[np.random.random(n) < 0.03, "city"] = np.nan

# 임시 CSV로 저장
csv_path = "examples/sample_data.csv"
df.to_csv(csv_path, index=False)
print(f"샘플 데이터 저장: {csv_path}")

# ── 2. f2a로 분석 ────────────────────────────────────────
import f2a

report = f2a.analyze(csv_path)

# 콘솔 요약 출력
report.show()

# 세부 통계 접근
print("\n=== 수치형 요약 ===")
print(report.stats.numeric_summary)

print("\n=== 상관 행렬 ===")
print(report.stats.correlation_matrix)

# HTML 리포트 생성
html_path = report.to_html("examples/output")
print(f"\nHTML 리포트 생성: {html_path}")
