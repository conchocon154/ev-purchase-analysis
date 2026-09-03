# What actually decides an EV purchase

A data analysis of the Kaggle competition
[Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)
(Playground Series S6E9) — 668,665 customer records, 17.46% of whom buy.

*Tiếng Việt: [README.vi.md](README.vi.md)*

[![CI](https://github.com/conchocon154/ev-purchase-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/conchocon154/ev-purchase-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Also published as a Kaggle notebook:**
[What actually decides an EV purchase](https://www.kaggle.com/code/minhngle/what-actually-decides-an-ev-purchase)
— the same analysis, self-contained and runnable against the competition data.

The competition asks for a probability. This is not a leaderboard write-up — it
asks the question a carmaker would ask instead: **which customers buy an electric
vehicle, what actually moves that decision, and who should we spend money
reaching?**

> **The data is synthetic.** Kaggle generated it from a model trained on a real
> EV adoption survey; feature distributions are close to, but not the same as,
> the original. Nothing here describes a real EV market, and
> [Limits](#limits-of-this-analysis) lists the specific places where the data
> behaves in ways a real market does not. Data licensed CC BY 4.0.

---

## What the analysis found

**1. Four variables carry the decision. The rest are noise dressed as signal.**

![Driver ranking](reports/charts/driver_ranking.png)

| Variable | Cramér's V | p-value | Verdict |
|---|---|---|---|
| Environmental concern | 0.497 | < 1e-300 | very strong |
| Subsidy available | 0.342 | < 1e-300 | very strong |
| Range anxiety | 0.116 | < 1e-300 | moderate |
| Home charging possible | 0.084 | < 1e-300 | weak |
| City type | 0.033 | 5e-162 | negligible |
| Current car type | 0.016 | 2e-37 | negligible |
| Number of cars owned | 0.009 | 6e-12 | negligible |
| Gender | 0.007 | **6.7e-08** | negligible |

Gender is *statistically significant* at p = 0.000000067 — and utterly
irrelevant, separating buyers by 0.53 percentage points. At 668,665 rows a
significance test will flag almost anything. **The p-value tells you a
difference is real; only the effect size tells you it matters.** Every table in
this repository reports both, because reporting the first alone is how an
analysis ends up recommending a gender-targeted campaign worth nothing.

**2. The interaction that isn't — and it changed the conclusion.**

The cross-tab looks emphatic. Among the most environmentally concerned
customers, 69.3% buy when a subsidy is available and 2.5% buy when it is not:

![Interaction](reports/charts/interaction.png)

The obvious reading is that the subsidy *multiplies* the effect of everything
else — that it is a gate, not a factor. That was the first conclusion drawn
here, and it was wrong.

A likelihood ratio test on the subsidy × concern interaction term returns
**LR = 7.2 on 9 degrees of freedom, p = 0.62**. McFadden's pseudo-R² is 0.4574
with the interaction and 0.4574 without it. The term buys nothing.

Checking directly: a logistic model with **main effects only** reproduces every
cell of that table to within 0.12 percentage points.

| Concern | Observed (no subsidy) | Predicted | Observed (subsidy) | Predicted |
|---|---|---|---|---|
| 1 | 0.02% | 0.01% | 0.94% | 0.94% |
| 3 | 0.30% | 0.27% | 18.66% | 18.68% |
| 5 | 2.48% | 2.60% | 69.33% | 69.29% |

The two effects are independent — on the odds scale. The dramatic-looking
pattern is what independent multiplicative effects always look like on the
*probability* scale when one group starts near zero: multiplying a 0.1% odds by
30 leaves you near zero, multiplying a 30% odds by 30 puts you near the ceiling.

This matters commercially. "Subsidies unlock environmental preference" and
"subsidies and preference each help, independently" imply different policies.
The first says target subsidised regions exclusively; the second says a
subsidy-free region with high-concern, high-income customers is still worth
addressing, just with lower expected conversion.

**3. Income is the strongest continuous driver, and it is a smooth gradient.**

![Income gradient](reports/charts/income_gradient.png)

From 6.3% in the lowest income sixth to 31.4% in the highest — a 25-point
spread with no threshold anywhere. There is no income level at which EVs
suddenly become affordable in this data; the propensity simply rises.

**4. Charging infrastructure does almost nothing, which should not be believed.**

| Variable | Spread across six bins |
|---|---|
| Annual income | 25.1 points |
| Daily commute | 5.2 points |
| Charging stations near home | 3.1 points |
| Age | 2.7 points |
| Charging stations near work | 1.7 points |

Two of these contradict what is known about real EV markets. Charging
availability near home is one of the most-cited determinants of EV adoption in
the published literature, and here it moves the purchase rate by 3 points — with
the *most* stations attached to a slightly *lower* rate. Longer commutes also
reduce purchase probability here, where a longer commute increases the fuel
saving that makes an EV pay back.

The `Home_Charging_Possible` flag does behave sensibly (19.6% against 12.7%). So
the generator modelled *whether you can charge at home*, but not *how many public
chargers are nearby*. Anyone using this data to argue about charger rollout would
be arguing from an artifact.

**5. A model you can explain gets within 0.3% of one you cannot.**

Five-fold cross-validated, scored on out-of-fold predictions:

| Model | ROC-AUC | Average precision | Brier |
|---|---|---|---|
| Logistic regression | 0.9381 | 0.7411 | 0.0736 |
| Gradient boosting | **0.9411** | **0.7533** | **0.0716** |

Accuracy is not reported, because predicting "nobody buys" scores 82.5% on this
data and describes a model that is useless. Average precision is reported
because it is sensitive to the 17% minority class, and the Brier score because
the output is used as a probability.

![Calibration](reports/charts/calibration.png)

Gradient boosting wins by 0.003 AUC. In a business setting that difference does
not pay for losing the ability to say *why* a customer scored the way they did,
so the logistic model is the one to ship. Its coefficients read directly as odds
ratios — for the numeric variables, per one standard deviation:

| Term | Odds ratio | Reading |
|---|---|---|
| Subsidy available: Yes | **91.4** | vs no subsidy |
| Range anxiety: Low | **29.3** | vs high anxiety |
| Environmental concern | **6.5** | per 1 SD (1.4 levels) |
| Annual income | 2.07 | per 1 SD (\$28,648) |
| Home charging: Yes | 1.26 | vs no home charging |
| Everything else | 0.91 – 1.07 | not worth acting on |

**6. What it is worth: contact the top 10% and reach 46% of all buyers.**

![Lift curve](reports/charts/lift_curve.png)

| Contact the top… | Buyers reached | Purchase rate in that group | Lift vs random |
|---|---|---|---|
| 10% | 45.7% | 79.9% | **4.6×** |
| 20% | 76.6% | 66.9% | 3.8× |
| 30% | 92.2% | 53.7% | 3.1× |
| 50% | 98.7% | 34.5% | 2.0× |

This is the number a marketing budget is actually set against. Contacting the
highest-scoring 30% of the list reaches 92% of everyone who would have bought:
the last 70% of the list contains 8% of the buyers, and chasing them costs more
than they return.

---

## What a carmaker should do with this

1. **Qualify on subsidy eligibility first.** It is the single largest term in
   the model, and it is knowable before any contact is made. Do not spend the
   acquisition budget where it does not apply — but do not write those regions
   off either: finding 2 says preference still operates there, at lower rates.
2. **Score, then cut the list at 30%.** Reaching 92% of buyers for 30% of the
   contact cost is the clearest operational win available here.
3. **Ask about range anxiety early.** An odds ratio of 29 between low and high
   anxiety, on a question a salesperson can simply ask, makes it the cheapest
   qualifying question available.
4. **Stop segmenting by gender, car ownership or current car type.** All three
   are statistically significant and commercially worthless.
5. **Do not use this data to plan charger placement.** See finding 4.

## Limits of this analysis

- **Synthetic data.** Generated by Kaggle from a model of a real survey. It
  supports conclusions about *this dataset*; extending them to a real market
  requires the real survey.
- **Observational, not causal.** Subsidy availability is not randomly assigned.
  The 91× odds ratio is an association. Subsidies plausibly cause purchases, but
  this data cannot separate that from subsidised regions differing in other ways.
- **A near-zero base rate is implausible.** 0.58% of unsubsidised customers buy.
  Real markets always contain buyers who are indifferent to incentives; a rate
  this absolute is a property of the generator.
- **No time dimension.** No purchase dates, so nothing about seasonality, price
  changes or the effect of a subsidy being introduced or withdrawn.
- **Test labels are unavailable.** Metrics come from cross-validation on the
  training set. No leaderboard score is claimed here.

## Method notes

- **Cross-validated out-of-fold predictions**, not a single hold-out: every row
  contributes to the estimate, and no row is scored by a model that saw it.
- **Wilson score intervals** on every rate. The high-range-anxiety group buys at
  0.14% on 2,194 rows; a normal approximation would put part of its interval
  below zero.
- **Cramér's V alongside every p-value**, for the reason in finding 1.
- **The interaction was tested, not eyeballed** — which is what overturned the
  original conclusion.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Needs a Kaggle API token and the competition rules accepted.
python scripts/fetch_data.py

python scripts/run_analysis.py        # tables, charts, reports/results.json
python -m pytest tests -q
```

`--sample 100000` fits the models on a subset for a faster run. `--no-model`
skips modelling entirely and produces the descriptive analysis in seconds.

The raw CSVs are not committed. The licence would permit it, but a repository is
easier to trust when its inputs come straight from the source.

## Layout

```
src/evstudy/
  data.py       loading, validation, column groupings
  segments.py   purchase rates with Wilson intervals, lift, cross-tabs
  stats.py      chi-square, Cramér's V, likelihood ratio test for interactions
  model.py      pipelines, cross-validated evaluation, calibration, decile lift
  charts.py     the figures
scripts/        fetch_data.py, run_analysis.py
reports/        results.json, tables/, charts/
tests/          statistical helpers checked against known values
```

## Tests

The statistical helpers are checked against values computed by hand or by an
independent method — a Wilson interval that is subtly wrong produces a report
that looks perfectly reasonable and is not.

```bash
python -m pytest tests -q
```

## Data and licence

Data: [Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9),
licensed CC BY 4.0, inspired by the EV adoption behaviour dataset. Code in this
repository: MIT, see [LICENSE](LICENSE).
