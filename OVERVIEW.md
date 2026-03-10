# German Electricity Market Prediction

**Ensimag IF -- Algorithmic Trading 2025 (Kaggle Competition)**

## Overview

This project develops a machine learning model to forecast short-term
price imbalances in the German electricity market.\
The objective is not simply prediction accuracy, but to generate signals
that drive a simulated trading strategy and maximise cumulative Profit &
Loss (PnL).

Electricity markets differ from traditional financial markets because
supply and demand must balance in real time. Forecasting errors in
renewable generation (wind/solar) or demand create imbalance prices,
which lead to exploitable short-horizon trading opportunities.

------------------------------------------------------------------------

## Competition Objective

For every 15-minute interval, we must predict the **spread**:

Spread = Imbalance Price − Day-Ahead Price

The competition then applies a fixed trading rule:

  Model Prediction   Trading Action
  ------------------ ------------------------------------------
  Spread ≥ 0         Buy electricity Day-Ahead (long 50 MW)
  Spread \< 0        Sell electricity Day-Ahead (short 50 MW)

The model itself does not choose trades --- it forecasts the spread, and
the platform evaluates profitability.

------------------------------------------------------------------------

## Evaluation Metric

Performance is measured by:

Cumulative Profit & Loss (€)

NOT RMSE or prediction error.

This means: - Correct directional prediction during stress events
matters far more than small numerical accuracy. - The task is closer to
**alpha modelling** than regression.

------------------------------------------------------------------------

## Data Used

Each observation represents a 15-minute market state:

  Variable   Meaning
  ---------- -------------------------
  wind       Wind generation (MW)
  solar      Solar generation (MW)
  load       Electricity demand (MW)
  date       Timestamp
  spread     Target variable

These variables capture the physical drivers of grid imbalance: -
Renewable intermittency\
- Demand fluctuations\
- Net system stress

------------------------------------------------------------------------

## Modelling Approach

### 1. Structural Feature Engineering

Economically meaningful variables were constructed:

renewable_gen = wind + solar\
renewable_pct = renewable_gen / load

net_demand = load − renewable_gen

This represents how much controllable generation must respond --- a key
driver of imbalance pricing.

------------------------------------------------------------------------

### 2. Market Dynamics Features

Electricity price dislocations are caused by rapid changes, not levels.

wind_ramp = Δ wind\
load_ramp = Δ load\
net_demand_ramp = Δ net_demand

Short-term stress indicators: - Rolling volatility (1-hour window) -
Lagged net demand (1h, 24h)

These approximate real-time grid instability that balancing markets must
correct.

------------------------------------------------------------------------

### 3. Time Structure

Electricity markets exhibit strong intraday seasonality.

Features added: - Hour of day - Day of week - Weekend indicator

------------------------------------------------------------------------

### 4. Model Choice

Gradient-boosted decision trees were used because they: - Capture
nonlinear regime behaviour - Model interactions between weather and
demand - Train efficiently on structured tabular time-series data

------------------------------------------------------------------------

## Validation Method

A time-series cross-validation scheme was used to mimic live trading:

Train on past → predict future → compute simulated PnL

This avoids look-ahead bias and reflects deployable performance.

------------------------------------------------------------------------

## Current Results

The model produces a steadily increasing simulated equity curve,
indicating capture of structural imbalance signals.

Latest leaderboard score:

€7,195,275 cumulative simulated PnL\
Rank ≈ Top 15 (public leaderboard)

------------------------------------------------------------------------

## What the Leaderboard Score Means

The score is the total profit (€) earned by the strategy over the test
period using our forecasts.

It is not a statistical accuracy metric.

A higher score means: - Better identification of grid
tightness/oversupply events - More profitable simulated trades under a
fixed execution rule

------------------------------------------------------------------------

## Key Takeaways

This project demonstrates: - Application of machine learning to a
real-time commodity market - Translating physical system behaviour into
predictive features - Designing models aligned with economic mechanisms
rather than generic ML - Evaluating performance using trading PnL
instead of prediction error

------------------------------------------------------------------------

## Summary

This work reframes electricity imbalance prediction as a quantitative
trading problem driven by renewable intermittency and real-time
supply--demand constraints, combining time-series modelling, feature
engineering, and economic intuition.