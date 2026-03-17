"""Dark theme configuration for matplotlib/seaborn charts."""

import matplotlib.pyplot as plt
import matplotlib as mpl

F2A_PALETTE = [
    "#58a6ff", "#3fb950", "#f97316", "#d2a8ff",
    "#79c0ff", "#56d364", "#e3b341", "#ff7b72",
    "#a5d6ff", "#7ee787", "#d29922", "#ffa198",
]

BG_COLOR = "#0d1117"
SURFACE_COLOR = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"


def apply_dark_theme() -> None:
    """Apply the f2a dark theme to matplotlib globally."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": SURFACE_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.facecolor": SURFACE_COLOR,
        "legend.edgecolor": GRID_COLOR,
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "savefig.facecolor": BG_COLOR,
    })
