"""Shared pieces of the generated reports: Markdown tables, number formatting and the figure style.

Colours are the reference categorical palette in its fixed slot order (methods) and its first three
slots, validated for all pairs, for the three wave classes; text wears text tokens, never series colours.
"""

import math
from collections.abc import Sequence
from typing import Any

import matplotlib

LABEL = {"head": "TimesFM head", "unet": "U-Net", "probe": "Linear probe", "neurokit_prominence": "NeuroKit2", "roundtrip": "Marks via decoder"}
SERIES = {"head": "#2a78d6", "unet": "#eb6834", "probe": "#1baf7a", "neurokit_prominence": "#eda100"}
WAVE = {"P": "#2a78d6", "QRS": "#eb6834", "T": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def apply_style() -> None:
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def fmt(x: float | None, digits: int = 1) -> str:
    """A number for a table; an en dash for missing."""
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
