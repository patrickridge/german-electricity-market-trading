# Ensimag IF - Algorithmic Trading 2025
## Quick Start Guide

### Competition Overview
**Goal:** Predict the spread between Day-Ahead and Imbalances electricity markets

**Spread = Imbalances Price - Day-Ahead Price**
- Spread > 0 → Energy deficit → Buy on Day-Ahead (long position)  
- Spread < 0 → Energy surplus → Sell on Day-Ahead (short position)

**Strategy:**
- If spread ≥ 0: Buy 50 MW on Day-Ahead
- If spread < 0: Sell 50 MW on Day-Ahead
- If Imbalances price > €1000: No position (too risky)

**Metric:** Cumulated Profit & Loss (€)

---

## Data Understanding

### Training Data (2020-2023)
- **date**: Timestamp (15-minute intervals)
- **wind**: Wind generation (MW)
- **solar**: Solar generation (MW)
- **load**: Electricity demand (MW)
- **spread**: Target variable (€/MWh)

### Test Data (2024)
- Same features except spread (which you need to predict)
- 24,138 predictions to make

### Key Statistics
- Mean spread: €0.89/MWh
- Std spread: €220.88/MWh
- Range: -€9,271 to +€15,781 (extreme values!)

---

## Key Insights from the Research Paper

The uploaded energy paper reveals critical market dynamics:

1. **Non-linear Cost Increase**: Balancing costs don't increase linearly with renewables—they explode at high penetration levels

2. **Steep Supply Curve**: At high demand, small imbalances cause massive price swings

3. **Forecast Errors**: Two sources of imbalances:
   - Weather forecast errors (wind/solar generation)
   - Load forecast errors (demand)

4. **Reduced Flexibility**: More renewables = less grid flexibility
   - Fossil fuels provide stabilization
   - Renewables are intermittent

5. **Critical Threshold**: Problems accelerate above ~30-35% renewable penetration

---

## Features That Matter

### Primary Drivers
1. **Renewable Penetration** = (wind + solar) / load
   - Include squared and cubed terms (non-linearity!)

2. **Net Position** = load - (wind + solar)
   - Positive = need conventional generation
   - Negative = excess renewables

3. **Time Patterns**
   - Hour of day (demand cycles)
   - Day of week (weekday vs weekend)
   - Season (heating/cooling demand)

### Advanced Features
4. **Volatility Proxies**
   - Rolling standard deviations
   - Rate of change indicators
   - Forecast error approximations

5. **Interaction Terms**
   - wind² , solar², load² (supply curve non-linearity)
   - wind × solar ratios
   - Renewable penetration³ (extreme value prediction)

---

## Step-by-Step Getting Started

### 1. Set Up on Kaggle
```
1. Go to Code tab
2. Find "ensimag-trading-if-2025"
3. Click "Copy & edit notebook"
4. You now have an editable Jupyter notebook!
```

### 2. Load and Explore Data
```python
import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('../input/ensimag-if-2025/train.csv', parse_dates=['date'])
test = pd.read_csv('../input/ensimag-if-2025/test.csv', parse_dates=['date'])

# Quick look
print(train.head())
print(train.describe())
```

### 3. Create Basic Features
```python
def create_features(df):
    df = df.copy()
    
    # Time features
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Renewable metrics
    df['renewable_pct'] = (df['wind'] + df['solar']) / df['load']
    df['renewable_pct2'] = df['renewable_pct'] ** 2
    df['renewable_pct3'] = df['renewable_pct'] ** 3
    
    # Net position
    df['net_demand'] = df['load'] - df['wind'] - df['solar']
    
    return df

train = create_features(train)
test = create_features(test)
```

### 4. Train a Simple Model
```python
from sklearn.ensemble import GradientBoostingRegressor

# Prepare data
features = ['hour', 'dayofweek', 'wind', 'solar', 'load', 
            'renewable_pct', 'renewable_pct2', 'renewable_pct3', 'net_demand']
X_train = train[features]
y_train = train['spread']
X_test = test[features]

# Train model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### 5. Create Submission
```python
submission = pd.DataFrame({
    'ID': test['ID'],
    'forecast': predictions
})

submission.to_csv('submission.csv', index=False)
print(submission.head())
```

---

## Ideas to Improve Your Score

### Short Term (Easy Wins)
1. **Add more time features**
   - Month, quarter, weekend indicator
   - Rush hour indicators (7-9am, 5-7pm)

2. **Rolling statistics**
   - Mean/std of last hour (4 periods)
   - Mean/std of last 24 hours

3. **Try different models**
   - XGBoost, LightGBM
   - CatBoost (handles categorical features well)
   - Ensemble multiple models

### Medium Term (More Effort)
4. **Feature engineering based on market dynamics**
   - Identify forecast error patterns
   - Capture supply curve non-linearity
   - Model seasonal effects

5. **Separate models for different conditions**
   - One model for high renewable periods
   - One model for low renewable periods
   - One model for extreme events

6. **Time series techniques**
   - Lag features (if past spreads matter)
   - ARIMA/SARIMA components
   - Prophet for seasonality

### Advanced (Competition Winners)
7. **Deep learning**
   - LSTM/GRU for sequence modeling
   - Attention mechanisms
   - Transformer architectures

8. **External data**
   - Weather forecasts (if allowed)
   - Gas prices (fuel costs affect spread)
   - Market events calendar

9. **Sophisticated ensembling**
   - Stacking
   - Blending
   - Meta-learners

---

## Common Pitfalls to Avoid

1. **Overfitting**: Don't use too many features without regularization
2. **Data leakage**: Don't use future information (e.g., rolling features must only look backward)
3. **Ignoring extreme values**: The -€9271 and +€15781 spreads matter!
4. **Treating time series as tabular**: Order matters in electricity markets
5. **Forgetting the domain**: Physics and economics drive the patterns

---

## Validation Strategy

Use **time series cross-validation**, not random splits!

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train and evaluate
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_val)
    # Calculate metrics...
```

---

## Resources

### Understanding the Market
- Read the uploaded energy paper (Energypaper.pdf)
- Key sections: Economic framework, Empirical results
- Focus on Figures 2-5, 9 (supply curves and balancing costs)

### Machine Learning
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Time Series Forecasting](https://www.kaggle.com/learn/time-series)
- [Feature Engineering](https://www.kaggle.com/learn/feature-engineering)

### Domain Knowledge
- Electricity market basics
- Renewable energy intermittency
- Grid balancing mechanisms

---

## Next Actions Checklist

- [ ] Copy the starter notebook on Kaggle
- [ ] Run baseline model
- [ ] Make first submission
- [ ] Analyze feature importance
- [ ] Read the energy research paper
- [ ] Add rolling statistics features
- [ ] Try XGBoost/LightGBM
- [ ] Implement time series CV
- [ ] Engineer renewable penetration features
- [ ] Ensemble top models
- [ ] Optimize hyperparameters

---

**Remember:** This is about learning! Focus on understanding why certain features work, not just blindly trying models.

Good luck! 🚀
