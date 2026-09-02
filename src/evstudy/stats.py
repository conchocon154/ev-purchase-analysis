"""Statistical tests behind the claims in the report.

A difference in group rates is not a finding until it survives a test, and a
test on 668,665 rows will call almost anything significant.  So every test here
is reported next to an effect size: with this much data the p-value tells you
the difference is real, and only the effect size tells you whether it matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import TARGET_BINARY


def cramers_v(table: np.ndarray) -> float:
    """Cramér's V: association strength between two categorical variables, 0-1."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    denominator = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator else 0.0


def association(frame: pd.DataFrame, column: str) -> dict:
    """Chi-square test of independence with the target, plus Cramér's V."""
    table = pd.crosstab(frame[column], frame[TARGET_BINARY]).to_numpy()
    chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
    return {
        "feature": column,
        "chi2": round(float(chi2), 1),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": round(cramers_v(table), 4),
    }


def association_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = [association(frame, column) for column in columns]
    table = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    # A conventional reading of Cramér's V, stated so the table is self-explaining.
    table["strength"] = pd.cut(
        table["cramers_v"],
        bins=[-0.01, 0.05, 0.10, 0.20, 0.30, 1.0],
        labels=["negligible", "weak", "moderate", "strong", "very strong"],
    )
    return table.reset_index(drop=True)


def _design(frame: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    return pd.get_dummies(frame[terms].astype(str), drop_first=True).astype(float)


def likelihood_ratio_test(frame: pd.DataFrame, base_terms: list[str],
                          interaction: tuple[str, str]) -> dict:
    """Does adding an interaction explain more than the two main effects alone?

    Fits both models by maximum likelihood and compares them with a likelihood
    ratio test.  Reading an interaction off a cross-tab is guesswork; this says
    whether the extra parameters earn their place.
    """
    import statsmodels.api as sm

    left, right = interaction
    y = frame[TARGET_BINARY].to_numpy()

    reduced_X = sm.add_constant(_design(frame, base_terms), has_constant="add")
    reduced = sm.Logit(y, reduced_X).fit(disp=0)

    combined = frame[left].astype(str) + " x " + frame[right].astype(str)
    full_frame = frame.assign(_interaction=combined)
    full_X = sm.add_constant(_design(full_frame, base_terms + ["_interaction"]),
                             has_constant="add")
    full = sm.Logit(y, full_X).fit(disp=0)

    statistic = 2 * (full.llf - reduced.llf)
    dof = int(full_X.shape[1] - reduced_X.shape[1])
    p_value = float(stats.chi2.sf(statistic, dof))
    # McFadden's pseudo-R^2 gain: how much of the remaining fit the term buys.
    null_ll = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).llf
    return {
        "interaction": f"{left} x {right}",
        "lr_statistic": round(float(statistic), 1),
        "dof": dof,
        "p_value": p_value,
        "pseudo_r2_reduced": round(1 - reduced.llf / null_ll, 4),
        "pseudo_r2_full": round(1 - full.llf / null_ll, 4),
    }


def rate_difference_ci(frame: pd.DataFrame, column: str, left: str, right: str) -> dict:
    """Difference in purchase rate between two levels, with a 95% interval."""
    subset = frame[frame[column].isin([left, right])]
    counts = subset.groupby(column, observed=True)[TARGET_BINARY].agg(["size", "sum"])
    n1, x1 = counts.loc[left, "size"], counts.loc[left, "sum"]
    n2, x2 = counts.loc[right, "size"], counts.loc[right, "sum"]
    p1, p2 = x1 / n1, x2 / n2
    difference = p1 - p2
    standard_error = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    margin = 1.959963984540054 * standard_error
    return {
        "comparison": f"{left} vs {right}",
        "rate_left_pct": round(100 * p1, 2),
        "rate_right_pct": round(100 * p2, 2),
        "difference_pp": round(100 * difference, 2),
        "ci_low_pp": round(100 * (difference - margin), 2),
        "ci_high_pp": round(100 * (difference + margin), 2),
        "risk_ratio": round(float(p1 / p2), 2) if p2 > 0 else float("inf"),
    }
