"""Visualization theme and style management."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


def _get_korean_font() -> str | None:
    """Find an available Korean font on the system."""
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
    """f2a visualization theme configuration.

    Attributes:
        palette: Seaborn color palette name.
        figsize: Default figure size.
        title_size: Title font size.
        label_size: Label font size.
        dpi: Output resolution.
        style: Seaborn style.
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
        """Apply the current theme to matplotlib/seaborn."""
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

        # Auto-configure Korean font
        korean_font = _get_korean_font()
        if korean_font:
            rc_params["font.family"] = korean_font
            rc_params["axes.unicode_minus"] = False  # Prevent minus sign rendering issues

        plt.rcParams.update(rc_params)

    def get_colors(self, n: int = 10) -> list[str]:
        """Return n colors from the palette."""
        return [str(c) for c in sns.color_palette(self.palette, n)]


# Default theme instance
DEFAULT_THEME = F2ATheme()
