"""Shared figure style for the studies in this folder."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

COLORS = {"HRP": "#1f5f8b", "CLA": "#c0392b", "IVP": "#6c9a5b", "1/N": "#8c8c8c"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "font.size": 9.5,
    }
)


def correlation_heatmap(ax: Axes, corr: np.ndarray, labels, title: str, show_labels: bool = True) -> None:
    """Correlation matrix on a fixed [-1, 1] diverging scale."""
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.grid(False)
    if show_labels:
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks, labels=[str(x) for x in labels], fontsize=8)
        ax.set_yticks(ticks, labels=[str(x) for x in labels], fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
