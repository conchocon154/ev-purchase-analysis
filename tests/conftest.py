import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def toy_frame() -> pd.DataFrame:
    """A small frame with a known structure, built without the Kaggle download.

    CI has no Kaggle credentials, so every test runs against data constructed
    here. The relationships are planted, so the helpers can be checked against
    an answer that is known in advance rather than against whatever the real
    data happens to contain.
    """
    rng = np.random.default_rng(11)
    n = 4000
    subsidy = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
    concern = rng.integers(1, 6, size=n).astype(float)
    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.5, 0.45, 0.05])
    income = rng.normal(85_000, 25_000, size=n).clip(30_000, 200_000)

    # Purchase odds driven only by subsidy, concern and income - gender is noise.
    logit = -6.0 + 3.0 * (subsidy == "Yes") + 0.9 * concern + 1.2e-5 * (income - 85_000)
    probability = 1 / (1 + np.exp(-logit))
    buys = rng.random(n) < probability

    return pd.DataFrame(
        {
            "Subsidy_Available": subsidy,
            "Environmental_Concern_Level": concern,
            "Gender": gender,
            "Annual_Income_USD": income,
            "Range_Anxiety_Level": rng.choice(["Low", "Medium", "High"], size=n, p=[.7, .25, .05]),
            "Home_Charging_Possible": rng.choice(["Yes", "No"], size=n),
            "Will_Buy_EV": np.where(buys, "Yes", "No"),
            "will_buy": buys.astype(int),
        }
    )
