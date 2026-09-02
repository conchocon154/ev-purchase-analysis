"""Figures for the report.

Labels are English in both language versions: the default matplotlib font has no
Vietnamese diacritics and would draw them as empty boxes.  The Vietnamese report
explains each figure in its own text instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1f2933"
ACCENT = "#0f766e"
WARM = "#c2410c"
MUTED = "#94a3b8"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def driver_ranking(ranking: pd.DataFrame, out: Path) -> Path:
    """How far apart the levels of each candidate driver sit."""
    data = ranking.sort_values("spread_pp")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colours = [ACCENT if v > 10 else MUTED for v in data["spread_pp"]]
    ax.barh(data["feature"], data["spread_pp"], color=colours)
    for index, value in enumerate(data["spread_pp"]):
        ax.text(value, index, f" {value:.1f}", va="center", fontsize=8, color=INK)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Spread between best and worst level (percentage points)", fontsize=8)
    ax.set_title("Only four variables separate buyers from non-buyers",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def rate_with_intervals(table: pd.DataFrame, label: str, out: Path) -> Path:
    """Purchase rate per level with its 95% interval drawn as an error bar."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    levels = table.iloc[:, 0].astype(str)
    rate = table["rate_pct"]
    lower = rate - table["ci_low_pct"]
    upper = table["ci_high_pct"] - rate
    ax.bar(levels, rate, color=ACCENT, width=0.6)
    ax.errorbar(levels, rate, yerr=[lower, upper], fmt="none",
                ecolor=INK, capsize=4, linewidth=1)
    for index, value in enumerate(rate):
        ax.text(index, value, f"{value:.1f}%", ha="center", va="bottom",
                fontsize=8, color=INK)
    _style(ax)
    ax.set_ylabel("Purchase rate (%)", fontsize=8)
    ax.set_title(f"Purchase rate by {label}", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def interaction_heatmap(table: pd.DataFrame, out: Path, *, xlabel: str, ylabel: str,
                        title: str) -> Path:
    """The interaction, drawn so the multiplication is visible rather than argued."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns.astype(str), fontsize=8)
    ax.set_yticks(range(table.shape[0]), table.index.astype(str), fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if value > values.max() * 0.55 else INK)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.tick_params(colors=INK)
    fig.colorbar(image, ax=ax, label="Purchase rate (%)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def income_gradient(table: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(len(table)), table["rate_pct"], marker="o", color=WARM, linewidth=1.8)
    ax.fill_between(range(len(table)), table["ci_low_pct"], table["ci_high_pct"],
                    color=WARM, alpha=0.15)
    ax.set_xticks(range(len(table)), table["bin"], rotation=20, ha="right", fontsize=7)
    _style(ax)
    ax.set_ylabel("Purchase rate (%)", fontsize=8)
    ax.set_title("Purchase rate rises steadily with income", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def calibration_plot(evaluations: list, out: Path) -> Path:
    """Predicted probability against observed rate: does 30% mean 30%?"""
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot([0, 1], [0, 1], linestyle=":", color=MUTED, linewidth=1, label="perfect")
    for evaluation, colour in zip(evaluations, [ACCENT, WARM, "#7c3aed"]):
        ax.plot(evaluation.calibration["predicted"], evaluation.calibration["observed"],
                marker="o", markersize=4, linewidth=1.5, color=colour, label=evaluation.name)
    _style(ax)
    ax.set_xlabel("Predicted probability", fontsize=8)
    ax.set_ylabel("Observed purchase rate", fontsize=8)
    ax.set_title("Calibration", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def lift_curve(lift: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(lift["top_pct"], lift["capture_pct"], marker="o", color=ACCENT,
            linewidth=1.8, label="model ranking")
    ax.plot(lift["top_pct"], lift["top_pct"], linestyle=":", color=MUTED,
            linewidth=1, label="random contact")
    _style(ax)
    ax.set_xlabel("Share of the list contacted (%)", fontsize=8)
    ax.set_ylabel("Share of buyers reached (%)", fontsize=8)
    ax.set_title("Contacting the highest-scoring customers first",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
