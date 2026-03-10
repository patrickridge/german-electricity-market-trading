# German Electricity Market — Spread Forecasting

**Ensimag IF · Algorithmic Trading 2025 · Kaggle Competition**

A machine learning model to forecast short-term price imbalances in the German electricity market. The objective is to generate trading signals that maximise cumulative Profit & Loss (PnL), not prediction accuracy.

> **Leaderboard result:** ~€7.2M cumulative PnL · Rank ≈ Top 15 (public)

---

## The Problem

Every 15 minutes, the German grid must balance real-time supply and demand. When renewable generation or demand is mis-forecast, the **Imbalance Price** diverges from the **Day-Ahead Price**. The task is to predict that divergence:

```
Spread = Imbalance Price − Day-Ahead Price
```

The platform applies a fixed trading rule based on the prediction:

| Predicted Spread | Action |
|---|---|
| ≥ 0 (deficit) | Buy 50 MW on Day-Ahead (long) |
| < 0 (surplus) | Sell 50 MW on Day-Ahead (short) |

**Score = total simulated profit in euros** over the 2024 test period.

---

## Key Insight

Electricity imbalance costs are non-linear. At high renewable penetration (>30–35%), small forecast errors cause extreme price swings due to a steep supply curve. The model captures this with polynomial renewable penetration features.

---

## Repository Structure

```
.
├── README.md
├── OVERVIEW.md                         # Detailed project overview and modelling approach
├── RESEARCH.md                         # Domain research: energy market economics
├── START.md                            # Quick start guide and feature ideas
├── improved_starter_notebook.py        # Clean Python implementation
├── ensimag-trading-if-2025-2.ipynb     # Full Jupyter notebook (EDA + model + CV)
├── baseline_submission.csv             # Baseline submission file
└── data/
    ├── train.csv                       # Training data 2020–2023 (~105k rows)
    ├── test.csv                        # Test data 2024 (24,138 rows)
    ├── imbalances.csv                  # Real imbalance prices for PnL evaluation
    └── sample.csv                      # Sample submission format
```

---

## Data

Each row is a 15-minute market observation:

| Column | Description |
|---|---|
| `date` | Timestamp |
| `wind` | Wind generation (MW) |
| `solar` | Solar generation (MW) |
| `load` | Electricity demand (MW) |
| `spread` | Target: Imbalance − Day-Ahead price (€/MWh) |

**Training:** 2020–2023 · **Test:** 2024
**Spread range:** −€9,271 to +€15,781 (extreme tail events matter)

---

## Modelling Approach

### Feature Engineering

```python
# Renewable penetration (with non-linear terms)
df['renewable_pct']  = (df['wind'] + df['solar']) / df['load']
df['renewable_pct2'] = df['renewable_pct'] ** 2
df['renewable_pct3'] = df['renewable_pct'] ** 3

# Net demand (controllable generation required)
df['net_demand'] = df['load'] - df['wind'] - df['solar']

# Ramp rates (rate of change drives imbalances)
df['wind_ramp']       = df['wind'].diff()
df['net_demand_ramp'] = df['net_demand'].diff()

# Time structure (intraday seasonality)
df['hour']       = df['date'].dt.hour
df['dayofweek']  = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
```

### Model

Gradient-boosted decision trees (scikit-learn `GradientBoostingRegressor`, also compatible with XGBoost/LightGBM/CatBoost). Tree-based methods handle the non-linear, regime-switching behaviour of electricity markets well.

### Validation

Time-series cross-validation (`TimeSeriesSplit`) — train on past, predict future — to avoid look-ahead bias and simulate realistic deployment.

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly
```

---

## Running

**Python script:**
```bash
python improved_starter_notebook.py
```

**Jupyter notebook:**
```bash
jupyter notebook ensimag-trading-if-2025-2.ipynb
```

> Note: The notebook uses Kaggle-style data paths (`../input/ensimag-if-2025/`). When running locally, update paths to `data/train.csv` etc.

---

## References

- ENTSO-E transparency platform — European electricity market data
- Energy economics research on balancing cost non-linearity at high renewable penetration (see `RESEARCH.md` for summary)
- Kaggle competition: [Ensimag IF 2025](https://www.kaggle.com/competitions/ensimag-if-2025)
