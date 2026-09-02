"""The interval and rate helpers are checked against values known in advance."""

import numpy as np
import pandas as pd
import pytest

from evstudy.segments import binned_rate, rank_drivers, rate_by, wilson_interval


def test_wilson_matches_a_published_value():
    # Classic worked example: 2 successes in 10 trials.
    low, high = wilson_interval(np.array([2]), np.array([10]))
    assert low[0] == pytest.approx(0.0567, abs=1e-3)
    assert high[0] == pytest.approx(0.5098, abs=1e-3)


def test_wilson_stays_inside_zero_and_one_for_extreme_rates():
    # 3 buyers in 2194 rows: a normal approximation would go below zero.
    low, high = wilson_interval(np.array([3]), np.array([2194]))
    assert low[0] > 0
    assert high[0] < 0.01


def test_wilson_interval_brackets_the_point_estimate():
    successes = np.array([1, 50, 999])
    total = np.array([10, 100, 1000])
    low, high = wilson_interval(successes, total)
    assert np.all(low <= successes / total)
    assert np.all(successes / total <= high)


def test_wilson_narrows_as_the_sample_grows():
    widths = []
    for n in (100, 1_000, 100_000):
        low, high = wilson_interval(np.array([n // 5]), np.array([n]))
        widths.append(high[0] - low[0])
    assert widths[0] > widths[1] > widths[2]


def test_rate_by_counts_and_rates_are_consistent(toy_frame):
    table = rate_by(toy_frame, "Subsidy_Available")
    assert table["n"].sum() == len(toy_frame)
    assert table["buyers"].sum() == toy_frame["will_buy"].sum()
    recomputed = 100 * table["buyers"] / table["n"]
    assert (recomputed - table["rate_pct"]).abs().max() < 0.01
    assert (table["ci_low_pct"] <= table["rate_pct"]).all()
    assert (table["rate_pct"] <= table["ci_high_pct"]).all()


def test_lift_is_relative_to_the_overall_rate(toy_frame):
    table = rate_by(toy_frame, "Subsidy_Available")
    overall = toy_frame["will_buy"].mean()
    for _, row in table.iterrows():
        assert row["lift"] == pytest.approx((row["buyers"] / row["n"]) / overall, abs=0.01)


def test_ordered_levels_are_not_sorted_by_rate(toy_frame):
    # Range anxiety must read Low, Medium, High - the meaningful order.
    table = rate_by(toy_frame, "Range_Anxiety_Level")
    assert list(table["Range_Anxiety_Level"]) == ["Low", "Medium", "High"]


def test_rank_drivers_puts_the_planted_signal_on_top(toy_frame):
    ranking = rank_drivers(
        toy_frame,
        ["Gender", "Subsidy_Available", "Environmental_Concern_Level", "Home_Charging_Possible"],
    )
    signal = {"Subsidy_Available", "Environmental_Concern_Level"}
    noise = {"Gender", "Home_Charging_Possible"}
    assert set(ranking.head(2)["feature"]) == signal
    # Both noise features are absent from the generator's logit, so their order
    # relative to each other is arbitrary - only their position below the signal
    # is guaranteed.
    worst_signal = ranking[ranking["feature"].isin(signal)]["spread_pp"].min()
    best_noise = ranking[ranking["feature"].isin(noise)]["spread_pp"].max()
    assert best_noise < worst_signal


def test_binned_rate_keeps_quantile_bins_in_order(toy_frame):
    table = binned_rate(toy_frame, "Annual_Income_USD", bins=5)
    lower_edges = [float(b.split(",")[0].lstrip("([")) for b in table["bin"]]
    assert lower_edges == sorted(lower_edges)
    assert table["n"].sum() == len(toy_frame)
