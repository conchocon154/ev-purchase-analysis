"""Load and validate the competition data.

The raw files are not in the repository — `scripts/fetch_data.py` downloads them
from Kaggle. Everything downstream goes through `load_train`, so the target
encoding and the column groupings are defined once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

TARGET = "Will_Buy_EV"
TARGET_BINARY = "will_buy"

NUMERIC = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

# Ordered levels, so tables and charts read low-to-high instead of alphabetically.
ORDERINGS = {
    "Range_Anxiety_Level": ["Low", "Medium", "High"],
    "Environmental_Concern_Level": [1.0, 2.0, 3.0, 4.0, 5.0],
    "Home_Charging_Possible": ["No", "Yes"],
    "Subsidy_Available": ["No", "Yes"],
}


class DataNotDownloaded(FileNotFoundError):
    pass


def _require(path: Path) -> Path:
    if not path.exists():
        raise DataNotDownloaded(
            f"{path.name} not found. Run: python scripts/fetch_data.py"
        )
    return path


def load_train(path: Path | None = None) -> pd.DataFrame:
    """Read train.csv and add the 0/1 target the analysis works with."""
    frame = pd.read_csv(_require(path or DATA / "train.csv"))
    if TARGET not in frame.columns:
        raise ValueError(f"expected a {TARGET!r} column, found {list(frame.columns)}")
    frame[TARGET_BINARY] = (frame[TARGET] == "Yes").astype(int)
    return frame


def load_test(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_require(path or DATA / "test.csv"))


def feature_columns() -> list[str]:
    return NUMERIC + CATEGORICAL


def describe_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """A one-look data-quality table: types, missingness, cardinality."""
    return pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing": frame.isna().sum(),
            "missing_pct": (100 * frame.isna().mean()).round(3),
            "distinct": frame.nunique(),
        }
    )
