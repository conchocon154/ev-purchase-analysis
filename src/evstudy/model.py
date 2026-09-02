"""Predictive models, evaluated the way an imbalanced problem has to be.

Only 17.5% of rows are buyers, so accuracy is worthless here — predicting "no"
for everyone scores 82.5%.  The metrics that matter are ROC-AUC, average
precision (the area under the precision-recall curve, which is sensitive to the
minority class), and calibration: if the model says 30%, do 30% of those people
actually buy?  A marketing budget is allocated on the probability, not on the
label, so a miscalibrated model misallocates money even at a high AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, TARGET_BINARY


@dataclass
class Evaluation:
    name: str
    roc_auc: float
    average_precision: float
    brier: float
    positive_rate: float
    calibration: pd.DataFrame = field(repr=False)
    predictions: np.ndarray = field(repr=False)

    def row(self) -> dict:
        return {
            "model": self.name,
            "roc_auc": round(self.roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "brier": round(self.brier, 5),
        }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
        ]
    )


def logistic_model() -> Pipeline:
    """Interpretable baseline: coefficients can be read as odds ratios."""
    return Pipeline(
        [
            ("prepare", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )


def logistic_with_interaction() -> Pipeline:
    """The same, plus the subsidy x environmental-concern term.

    Included because the cross-tab says the effect of environmental concern is
    almost entirely conditional on subsidy eligibility; a model without the term
    has to average those two very different worlds into one coefficient.
    """
    return Pipeline(
        [
            ("prepare", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )


def gradient_boosting() -> Pipeline:
    """Finds interactions on its own; used to bound what the features can give."""
    return Pipeline(
        [
            ("prepare", build_preprocessor()),
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                early_stopping=True, validation_fraction=0.1, random_state=42)),
        ]
    )


def add_interaction(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialise the interaction as a categorical column the pipeline can use."""
    out = frame.copy()
    out["Subsidy_x_Concern"] = (
        out["Subsidy_Available"].astype(str) + "_" +
        out["Environmental_Concern_Level"].astype(str)
    )
    return out


def evaluate(name: str, pipeline: Pipeline, frame: pd.DataFrame,
             features: list[str], folds: int = 5, seed: int = 42) -> Evaluation:
    """Cross-validated out-of-fold probabilities, then score them.

    Scoring on out-of-fold predictions rather than a single hold-out means every
    row contributes to the estimate and no row is scored by a model that saw it.
    """
    X = frame[features]
    y = frame[TARGET_BINARY].to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        pipeline, X, y, cv=splitter, method="predict_proba", n_jobs=1
    )[:, 1]

    fraction_positive, mean_predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
    calibration = pd.DataFrame(
        {"predicted": mean_predicted.round(4), "observed": fraction_positive.round(4)}
    )
    return Evaluation(
        name=name,
        roc_auc=float(roc_auc_score(y, probabilities)),
        average_precision=float(average_precision_score(y, probabilities)),
        brier=float(brier_score_loss(y, probabilities)),
        positive_rate=float(y.mean()),
        calibration=calibration,
        predictions=probabilities,
    )


def odds_ratios(pipeline: Pipeline, features: list[str]) -> pd.DataFrame:
    """Read a fitted logistic pipeline as multiplicative effects on the odds."""
    prepare = pipeline.named_steps["prepare"]
    names = list(prepare.get_feature_names_out())
    coefficients = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({"term": names, "coefficient": coefficients})
    table["odds_ratio"] = np.exp(table["coefficient"]).round(3)
    table["direction"] = np.where(table["coefficient"] > 0, "increases", "decreases")
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).drop(
        columns="abs_coefficient").reset_index(drop=True)


def lift_by_decile(probabilities: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """If you could only contact 10% of the list, how much better than random?

    This is the form the model's value takes in a marketing plan, and it is
    readable by people who will never look at an ROC curve.
    """
    order = np.argsort(-probabilities)
    ranked = y[order]
    n = len(ranked)
    baseline = ranked.mean()
    rows = []
    for decile in range(1, 11):
        cutoff = int(round(n * decile / 10))
        segment = ranked[:cutoff]
        rows.append(
            {
                "top_pct": decile * 10,
                "contacted": cutoff,
                "buyers_captured": int(segment.sum()),
                "capture_pct": round(100 * segment.sum() / ranked.sum(), 1),
                "rate_in_segment_pct": round(100 * segment.mean(), 2),
                "lift_vs_random": round(segment.mean() / baseline, 2),
            }
        )
    return pd.DataFrame(rows)
