"""시각화 테마 및 스타일 관리."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


def _get_korean_font() -> str | None:
    """시스템에서 사용 가능한 한글 폰트를 찾습니다."""
    system = platform.system()
    candidates: list[str] = []

    if system == "Windows":
        candidates = ["Malgun Gothic", "맑은 고딕", "NanumGothic", "NanumBarunGothic"]
    elif system == "Darwin":
        candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]
    else:
        candidates = ["NanumGothic", "NanumBarunGothic", "UnDotum", "Noto Sans CJK KR"]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return None


@dataclass
class F2ATheme:
    """f2a 시각화 테마 설정.

    Attributes:
        palette: seaborn 컬러 팔레트 이름.
        figsize: 기본 Figure 크기.
        title_size: 제목 폰트 크기.
        label_size: 라벨 폰트 크기.
        dpi: 출력 해상도.
        style: seaborn 스타일.
    """

    palette: str = "husl"
    figsize: tuple[float, float] = (10, 6)
    title_size: int = 14
    label_size: int = 11
    dpi: int = 100
    style: str = "whitegrid"
    context: str = "notebook"
    font_scale: float = 1.0
    _colors: list[str] = field(default_factory=list)

    def apply(self) -> None:
        """현재 테마를 matplotlib/seaborn에 적용합니다."""
        sns.set_theme(
            style=self.style,
            context=self.context,
            font_scale=self.font_scale,
            palette=self.palette,
        )

        rc_params: dict = {
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
        }

        # 한글 폰트 자동 설정
        korean_font = _get_korean_font()
        if korean_font:
            rc_params["font.family"] = korean_font
            rc_params["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

        plt.rcParams.update(rc_params)

    def get_colors(self, n: int = 10) -> list[str]:
        """팔레트에서 n개의 색상을 반환합니다."""
        return [str(c) for c in sns.color_palette(self.palette, n)]


# 기본 테마 인스턴스
DEFAULT_THEME = F2ATheme()
