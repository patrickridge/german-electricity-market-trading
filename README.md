# German Electricity Market — Spread Forecasting

**Ensimag IF · Algorithmic Trading 2025 · [Kaggle Competition](https://www.kaggle.com/competitions/ensimag-if-2025)**

Forecasting the spread between the German day-ahead and real-time imbalance electricity markets, scored on simulated trading PnL rather than prediction error.

> **Result: €18,950,726 cumulative PnL · 6th of 46 teams · 9.5× the competition's naive baseline**

---

## The Problem

Every 15 minutes the German grid settles the difference between what generators promised on the day-ahead market and what was actually delivered. When renewable output or demand is mis-forecast, the **imbalance price** diverges from the **day-ahead price**:

```
spread = imbalance price − day-ahead price
```

- **spread > 0** — grid is tighter than expected (deficit); expensive peaking plant is called, price rises
- **spread < 0** — grid is looser than expected (surplus); excess generation, price falls

The platform applies a fixed rule to the forecast and reports the resulting PnL:

| predicted spread | action |
|---|---|
| ≥ 0 | buy 50 MW day-ahead (long) |
| < 0 | sell 50 MW day-ahead (short) |
| any, when \|imbalance price\| > €1000 | no position (too risky) |

**Only the sign of the forecast is used.** The magnitude is discarded.

---

## Data

15-minute observations of `wind`, `solar`, `load` (all MW) and `spread` (€/MWh).

| | rows | period |
|---|---|---|
| train | 140,157 | 2020-01-01 → 2023-12-31 |
| test | 24,138 | 2024-01-01 → 2024-09-08 |

The target is close to a coin flip (P(spread ≥ 0) = 51.0%) with extreme tails: median €3.23, but a range of −€9,271 to +€15,781. Competition rules permit no external data.

---

## Approach

### Train the objective that is actually scored

The scorer reads only the sign, so the model classifies `sign(spread)` with `sample_weight = |spread|` rather than regressing the spread level under MSE.

Weighting by `|spread|` places the 0.5 probability contour exactly at `E[spread] = 0`, which is the PnL-optimal decision boundary — so no threshold tuning is needed. This was tested: the empirically optimal threshold came out at 0.490, and tuning it was worth +0.4%.

### Exclude rows the scorer discards

Rows where `|imbalance| > 1000` are never traded, but because weight is proportional to `|spread|` they were absorbing **7.1% of total training weight** while being 0.41% of rows. Zeroing them was worth +€1.98M on held-out folds and +€1.17M on the leaderboard.

### Features

Net demand (`load − wind − solar`) and its lag, ramp, rolling-mean and volatility structure, renewable penetration with polynomial terms, and calendar features. Recency-decayed sample weights (730-day half-life) to track the upward drift in renewable penetration between training and test periods.

### Validation

Walk-forward (`TimeSeriesSplit`) throughout. Hyperparameters and feature sets are selected on early folds and re-scored on later folds that never influenced the choice.

---

## What Worked and What Didn't

Every change below was measured on held-out walk-forward folds and then checked against the leaderboard.

| change | held-out | leaderboard |
|---|---|---|
| sign objective + previously unused features | large | **+€7.6M** |
| recency weighting + expanded lags | +€1.5–4.7M | **+€3.5M** |
| drop untradable rows from training weights | +€1.98M | **+€1.17M** |
| hyperparameter tuning | −€1.9M | −€197k |
| 10-seed averaging | +€1.76M | −€483k |
| cross-family blend (HistGB + ExtraTrees + RF) | +€897k | −€302k |
| regime-relative / cyclical features | ~neutral | not submitted |

Two findings worth noting:

**A nested search is not optional.** Ranking hyperparameter configs on all folds made a config look best that then *lost* €1.9M on folds it hadn't influenced — and the leaderboard confirmed it, scoring 14,553,020 against 14,750,344 for the untuned model. A flat search would have shipped a worse model with a confident number attached.

**Measure your noise floor before trusting an effect.** Seed-to-seed variance is ±€1.4M here. A single-seed test suggested regime features cost €3.57M; a matched-seed rerun showed them roughly neutral (+€591k). Every structural change measured well above that floor transferred to the leaderboard; both ensembling changes measured near it went negative.

---

## Repository

```
├── sign_classifier_model.py        # core model, walk-forward PnL, ablation
├── final_model.py                  # best submission (v5)
├── tune_sign_classifier.py         # nested hyperparameter search — negative result
├── experiment_recency_lags.py      # recency weighting + expanded lags
├── experiment_regime_features.py   # regime-relative features — rejected
├── final_model_blend.py            # cross-family blend — rejected on leaderboard
├── improved_starter_notebook.py    # cleaned starter script
├── ensimag-trading-if-2025-2.ipynb # original notebook
├── RESEARCH.md                     # domain notes on balancing-cost economics
├── START.md                        # competition quick-start
└── data/                           # train, test, imbalances, sample
```

## Running

```bash
pip install numpy pandas scikit-learn
python sign_classifier_model.py   # walk-forward evaluation + submission
python final_model.py             # best-scoring configuration
```

---

## A Note on RESEARCH.md

`RESEARCH.md` summarises an academic paper arguing that balancing costs rise non-linearly above ~30–35% renewable penetration, and the original model was built around polynomial penetration features on that basis.

Tested directly on this dataset, that claim does not hold for the spread. Bucketing training rows by renewable penetration, mean `|spread|` runs 106 / 100 / 94 / 102 / 108 / 115 €/MWh across `<15%` / `15–30%` / `30–35%` / `35–50%` / `50–70%` / `>70%` — a shallow U, with no acceleration at the claimed threshold.

The reconciliation is that the paper measures balancing *cost* (volume × price, a system-wide aggregate) while the competition target is the spread (a per-interval price differential). Cost can grow sharply through volume while the per-MWh gap stays flat.

What *does* hold is directional: P(spread ≥ 0) falls steadily from 55.8% to 44.8% across those same buckets. Since the trading rule reads only the sign, that tilt is the monetizable signal — and it is why `solar` dominates the feature importances.
