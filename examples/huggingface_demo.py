"""HuggingFace 데이터셋 분석 예제.

이 스크립트는 HuggingFace Hub의 데이터셋을 f2a로 분석하는 방법을 보여줍니다.

Prerequisites:
    pip install f2a[hf]

Usage:
    python examples/huggingface_demo.py
"""

import f2a

# ── HuggingFace 데이터셋 분석 ────────────────────────────
# hf:// 프리픽스를 사용하여 HuggingFace 데이터셋을 로딩합니다.

print("=== HuggingFace 데이터셋 분석 ===\n")

# 예시: IMDB 영화 리뷰 데이터셋
# (실행하려면 `datasets` 패키지가 필요합니다)
try:
    report = f2a.analyze("hf://imdb", split="train")
    report.show()

    # HTML 리포트 생성
    html_path = report.to_html("examples/output")
    print(f"\nHTML 리포트 생성: {html_path}")

except Exception as e:
    print(f"HuggingFace 로딩 실패: {e}")
    print("'datasets' 패키지를 설치하세요: pip install f2a[hf]")
