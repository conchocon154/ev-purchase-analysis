"""Build a competition submission.

The analysis in the README deliberately favours the explainable model. A
submission is judged on ROC-AUC alone, so the choice here is made on measured
out-of-fold score rather than on interpretability - and the candidates are
compared before one is picked, so the choice is recorded rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from evstudy.data import CATEGORICAL, NUMERIC, TARGET_BINARY, load_test, load_train  # noqa: E402
from evstudy.model import build_preprocessor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline([("prep", build_preprocessor()),
                              ("model", LogisticRegression(max_iter=1000, n_jobs=-1))]),
        "gbm": Pipeline([("prep", build_preprocessor()),
                         ("model", HistGradientBoostingClassifier(
                             max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                             l2_regularization=1.0, early_stopping=True,
                             validation_fraction=0.1, random_state=seed))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train = load_train()
    test = load_test()
    features = NUMERIC + CATEGORICAL
    X, y = train[features], train[TARGET_BINARY].to_numpy()

    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    oof, scores = {}, {}
    for name, pipeline in candidates(args.seed).items():
        oof[name] = cross_val_predict(pipeline, X, y, cv=splitter,
                                      method="predict_proba", n_jobs=1)[:, 1]
        scores[name] = roc_auc_score(y, oof[name])
        print(f"  {name:22} out-of-fold ROC-AUC {scores[name]:.5f}")

    # A rank-average blend: the two models make different errors, and averaging
    # ranks rather than probabilities avoids one model's sharper distribution
    # dominating the other's.
    ranks = sum(pd.Series(p).rank(pct=True).to_numpy() for p in oof.values()) / len(oof)
    scores["blend"] = roc_auc_score(y, ranks)
    print(f"  {'blend (rank average)':22} out-of-fold ROC-AUC {scores['blend']:.5f}")

    best = max(scores, key=scores.get)
    print(f"\nselected: {best} ({scores[best]:.5f})")

    fitted = {name: pipeline.fit(X, y) for name, pipeline in candidates(args.seed).items()}
    predictions = {name: model.predict_proba(test[features])[:, 1]
                   for name, model in fitted.items()}
    if best == "blend":
        final = sum(pd.Series(p).rank(pct=True).to_numpy()
                    for p in predictions.values()) / len(predictions)
    else:
        final = predictions[best]

    submission = pd.DataFrame({"id": test["id"], "Will_Buy_EV": final})
    submission.to_csv(args.out, index=False)

    print(f"\n{len(submission):,} rows -> {args.out}")
    print(f"  prediction range {final.min():.4f} to {final.max():.4f}, "
          f"mean {final.mean():.4f}")
    print(f"  training set positive rate {y.mean():.4f}")


if __name__ == "__main__":
    main()
