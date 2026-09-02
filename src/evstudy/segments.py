"""Purchase rates by segment, with intervals rather than bare percentages.

A rate quoted without an interval invites the reader to treat 19.34% and 18.09%
as different when the underlying counts may not support it.  Every rate here
carries a Wilson score interval, which behaves correctly for the small and very
lopsided groups this dataset contains (the High range-anxiety group buys at
0.14% on 2,194 rows — a normal approximation would put part of its interval
below zero).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ORDERINGS, TARGET_BINARY

Z_95 = 1.959963984540054


def wilson_interval(successes: np.ndarray, total: np.ndarray, z: float = Z_95):
    """Wilson score interval for a binomial proportion."""
    successes = np.asarray(successes, dtype=float)
    total = np.asarray(total, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = np.divide(successes, total, out=np.zeros_like(successes), where=total > 0)
        denominator = 1 + z**2 / total
        centre = phat + z**2 / (2 * total)
        spread = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
        low = (centre - spread) / denominator
        high = (centre + spread) / denominator
    return np.clip(low, 0, 1), np.clip(high, 0, 1)


def rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Purchase rate for each level of `column`, with counts and a 95% interval."""
    grouped = frame.groupby(column, observed=True)[TARGET_BINARY].agg(["size", "sum"])
    grouped.columns = ["n", "buyers"]
    low, high = wilson_interval(grouped["buyers"].to_numpy(), grouped["n"].to_numpy())
    grouped["rate_pct"] = (100 * grouped["buyers"] / grouped["n"]).round(2)
    grouped["ci_low_pct"] = (100 * low).round(2)
    grouped["ci_high_pct"] = (100 * high).round(2)

    baseline = frame[TARGET_BINARY].mean()
    grouped["lift"] = (grouped["buyers"] / grouped["n"] / baseline).round(2)

    order = ORDERINGS.get(column)
    if order is not None:
        present = [level for level in order if level in grouped.index]
        grouped = grouped.loc[present]
    else:
        grouped = grouped.sort_values("rate_pct", ascending=False)
    return grouped.reset_index()


def spread_of(frame: pd.DataFrame, column: str) -> float:
    """How many percentage points separate the best and worst level."""
    table = rate_by(frame, column)
    return float(table["rate_pct"].max() - table["rate_pct"].min())


def rank_drivers(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Order the candidate drivers by how far apart their levels sit.

    Deliberately a crude measure - it says nothing about significance or about
    what survives when the other variables are held constant.  It is a triage
    step to decide what deserves a closer look, and `stats.py` does the testing.
    """
    rows = []
    for column in columns:
        table = rate_by(frame, column)
        rows.append(
            {
                "feature": column,
                "levels": len(table),
                "min_rate_pct": table["rate_pct"].min(),
                "max_rate_pct": table["rate_pct"].max(),
                "spread_pp": round(table["rate_pct"].max() - table["rate_pct"].min(), 2),
                "max_lift": table["lift"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("spread_pp", ascending=False).reset_index(drop=True)


def binned_rate(frame: pd.DataFrame, column: str, bins: int = 6) -> pd.DataFrame:
    """Purchase rate across quantile bins of a continuous variable."""
    binned = frame.assign(_bin=pd.qcut(frame[column], bins, duplicates="drop"))
    table = rate_by(binned, "_bin")
    table = table.rename(columns={"_bin": "bin"})
    table["bin"] = table["bin"].astype(str)
    # rate_by sorts unordered columns by rate; quantile bins must stay in order.
    return table.sort_values("bin", key=lambda s: s.map(
        {str(c): i for i, c in enumerate(binned["_bin"].cat.categories)}
    )).reset_index(drop=True)


def crosstab_rate(frame: pd.DataFrame, index: str, column: str) -> pd.DataFrame:
    """Purchase rate for every combination of two variables."""
    table = frame.pivot_table(index=index, columns=column,
                              values=TARGET_BINARY, aggfunc="mean") * 100
    for axis_name, axis in (("index", index), ("columns", column)):
        order = ORDERINGS.get(axis)
        if order is None:
            continue
        labels = table.index if axis_name == "index" else table.columns
        present = [level for level in order if level in labels]
        table = table.loc[present] if axis_name == "index" else table[present]
    return table.round(2)
