"""Run the full study and write every table, figure and number the report cites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from evstudy import charts  # noqa: E402
from evstudy.data import (  # noqa: E402
    CATEGORICAL, NUMERIC, TARGET_BINARY, describe_quality, load_train,
)
from evstudy.model import (  # noqa: E402
    evaluate, gradient_boosting, lift_by_decile, logistic_model, odds_ratios,
)
from evstudy.segments import (  # noqa: E402
    binned_rate, crosstab_rate, rank_drivers, rate_by,
)
from evstudy.stats import association_table, likelihood_ratio_test  # noqa: E402

DRIVERS = CATEGORICAL + ["Environmental_Concern_Level", "Number_of_Cars_Owned"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--sample", type=int, default=None,
                        help="fit the models on a random subset (for a quick run)")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tables = args.out / "tables"
    figures = args.out / "charts"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    frame = load_train()
    print(f"train: {len(frame):,} rows, purchase rate {100 * frame[TARGET_BINARY].mean():.2f}%")

    results: dict = {
        "rows": int(len(frame)),
        "purchase_rate_pct": round(100 * float(frame[TARGET_BINARY].mean()), 2),
    }

    quality = describe_quality(frame)
    quality.to_csv(tables / "data_quality.csv")
    results["missing_values_total"] = int(quality["missing"].sum())

    ranking = rank_drivers(frame, DRIVERS)
    ranking.to_csv(tables / "driver_ranking.csv", index=False)
    charts.driver_ranking(ranking, figures / "driver_ranking.png")
    results["driver_ranking"] = ranking.to_dict("records")

    associations = association_table(frame, DRIVERS)
    associations.to_csv(tables / "associations.csv", index=False)
    results["associations"] = associations.assign(
        p_value=associations["p_value"].map(lambda v: f"{v:.3e}")
    ).to_dict("records")

    for column, label in [("Subsidy_Available", "subsidy availability"),
                          ("Environmental_Concern_Level", "environmental concern"),
                          ("Range_Anxiety_Level", "range anxiety")]:
        table = rate_by(frame, column)
        table.to_csv(tables / f"rate_{column}.csv", index=False)
        charts.rate_with_intervals(table, label, figures / f"rate_{column}.png")

    income = binned_rate(frame, "Annual_Income_USD")
    income.to_csv(tables / "rate_income_bins.csv", index=False)
    charts.income_gradient(income, figures / "income_gradient.png")
    results["income_low_pct"] = float(income["rate_pct"].iloc[0])
    results["income_high_pct"] = float(income["rate_pct"].iloc[-1])

    interaction = crosstab_rate(frame, "Environmental_Concern_Level", "Subsidy_Available")
    interaction.to_csv(tables / "interaction_subsidy_concern.csv")
    charts.interaction_heatmap(
        interaction, figures / "interaction.png",
        xlabel="Subsidy available", ylabel="Environmental concern level",
        title="Purchase rate by concern and subsidy")
    results["interaction_table"] = interaction.to_dict()

    lr = likelihood_ratio_test(
        frame,
        ["Subsidy_Available", "Environmental_Concern_Level",
         "Range_Anxiety_Level", "Home_Charging_Possible"],
        ("Subsidy_Available", "Environmental_Concern_Level"),
    )
    results["interaction_test"] = lr
    print(f"interaction LR test: p = {lr['p_value']:.3f}")

    if not args.no_model:
        fitting = frame.sample(args.sample, random_state=args.seed) if args.sample else frame
        features = NUMERIC + CATEGORICAL
        evaluations = []
        for name, pipeline in [("logistic regression", logistic_model()),
                               ("gradient boosting", gradient_boosting())]:
            print(f"  evaluating {name} on {len(fitting):,} rows ...")
            evaluations.append(evaluate(name, pipeline, fitting, features, seed=args.seed))

        scores = pd.DataFrame([e.row() for e in evaluations])
        scores.to_csv(tables / "model_scores.csv", index=False)
        results["model_scores"] = scores.to_dict("records")
        print(scores.to_string(index=False))

        charts.calibration_plot(evaluations, figures / "calibration.png")

        best = max(evaluations, key=lambda e: e.roc_auc)
        lift = lift_by_decile(best.predictions, fitting[TARGET_BINARY].to_numpy())
        lift.to_csv(tables / "lift_by_decile.csv", index=False)
        charts.lift_curve(lift, figures / "lift_curve.png")
        results["best_model"] = best.name
        results["lift"] = lift.to_dict("records")

        fitted = logistic_model().fit(fitting[features], fitting[TARGET_BINARY])
        coefficients = odds_ratios(fitted, features)
        coefficients.to_csv(tables / "odds_ratios.csv", index=False)
        results["odds_ratios"] = coefficients.head(12).to_dict("records")

    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults -> {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
