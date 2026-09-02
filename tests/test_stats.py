"""Association measures and the interaction test."""

import numpy as np
import pytest

from evstudy.stats import association_table, cramers_v, rate_difference_ci


def test_cramers_v_is_zero_for_independent_variables():
    table = np.array([[100, 100], [200, 200]])
    assert cramers_v(table) == pytest.approx(0.0, abs=1e-9)


def test_cramers_v_is_one_for_a_perfect_association():
    table = np.array([[500, 0], [0, 500]])
    assert cramers_v(table) == pytest.approx(1.0, abs=1e-9)


def test_cramers_v_stays_in_range(toy_frame):
    table = association_table(toy_frame, ["Gender", "Subsidy_Available"])
    assert (table["cramers_v"] >= 0).all()
    assert (table["cramers_v"] <= 1).all()


def test_association_ranks_signal_above_noise(toy_frame):
    table = association_table(
        toy_frame, ["Gender", "Subsidy_Available", "Environmental_Concern_Level"]
    )
    assert table.iloc[0]["feature"] in {"Subsidy_Available", "Environmental_Concern_Level"}
    gender = table[table["feature"] == "Gender"].iloc[0]
    assert gender["cramers_v"] < 0.05
    assert gender["strength"] == "negligible"


def test_rate_difference_interval_brackets_the_difference(toy_frame):
    result = rate_difference_ci(toy_frame, "Subsidy_Available", "Yes", "No")
    assert result["ci_low_pp"] <= result["difference_pp"] <= result["ci_high_pp"]
    assert result["difference_pp"] > 0          # subsidy was planted as positive
    assert result["risk_ratio"] > 1


def test_interaction_test_finds_nothing_when_effects_are_additive(toy_frame):
    """The toy data has no interaction by construction, so the test must say so.

    This is the guard on the finding that overturned the original conclusion:
    a test that reports an interaction in additive data would have been reporting
    an artefact.
    """
    from evstudy.stats import likelihood_ratio_test

    result = likelihood_ratio_test(
        toy_frame,
        ["Subsidy_Available", "Environmental_Concern_Level"],
        ("Subsidy_Available", "Environmental_Concern_Level"),
    )
    assert result["p_value"] > 0.01
    assert result["pseudo_r2_full"] >= result["pseudo_r2_reduced"] - 1e-9
