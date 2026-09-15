# Cell 1: Imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

# Cell 2: Load and explore data
# Load data
train = pd.read_csv('/kaggle/input/ensimag-if-2025/train.csv', parse_dates=['date'], index_col='date')
test = pd.read_csv('/kaggle/input/ensimag-if-2025/test.csv', parse_dates=['date'])

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns.tolist()}")
print(f"\nFirst few rows:")
print(train.head())

print(f"\nSpread statistics:")
print(train['spread'].describe())

# Cell 3: Quick visualization
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['spread'], mode='lines', name='spread'))
fig.update_layout(title='Spread Over Time', xaxis_title='Date', yaxis_title='Spread (€/MWh)')
fig.show()

# Cell 4: Feature Engineering
# Create features based on energy market dynamics
def create_features(df):
    """Add features to the dataframe"""
    df = df.copy()
    
    # Temporal features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Renewable features (KEY INSIGHT FROM PAPER!)
    df['renewable_gen'] = df['wind'] + df['solar']
    df['renewable_pct'] = df['renewable_gen'] / df['load']
    df['renewable_pct2'] = df['renewable_pct'] ** 2  # Non-linearity
    df['renewable_pct3'] = df['renewable_pct'] ** 3  # Extreme values
    
    # Net position (demand minus renewables)
    df['net_demand'] = df['load'] - df['renewable_gen']
    df['net_demand_ratio'] = df['net_demand'] / df['load']
    
    return df

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

print(f"New features created: {[c for c in train_fe.columns if c not in train.columns]}")

# Cell 5: Prepare data for modeling
# Features to use
feature_cols = ['hour', 'dayofweek', 'month', 'is_weekend',
                'wind', 'solar', 'load',
                'renewable_gen', 'renewable_pct', 'renewable_pct2', 'renewable_pct3',
                'net_demand', 'net_demand_ratio']

x_train = train_fe[feature_cols].fillna(0)
y_train = train_fe['spread']

x_test = test_fe.set_index('ID')[feature_cols].fillna(0)

print(f"Training with {len(feature_cols)} features")
print(f"X_train shape: {x_train.shape}")
print(f"X_test shape: {x_test.shape}")

# Cell 6: Train improved model
# Use Gradient Boosting instead of Linear Regression
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(x_train, y_train)
print("Model trained!")

# Feature importance
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_imp.head(10))

# Cell 7: Make predictions and create submission
# Predict on test set
pred = model.predict(x_test)

# Create submission dataframe
submission = pd.DataFrame({
    'ID': x_test.index,
    'forecast': pred
})

# Save submission file
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✓ File submission.csv created")
print(f"\nFirst 10 predictions:")
print(submission.head(10))
print(f"\nPrediction stats:")
print(f"Mean: {pred.mean():.2f} €/MWh")
print(f"Std: {pred.std():.2f} €/MWh")

# Cell 8: OPTIONAL - Cross-validation to estimate performance
print("Running walk-forward cross-validation to estimate performance...")

# TimeSeriesSplit, not KFold: each fold may only train on data preceding its
# validation window, otherwise the model scores 2020 using 2023 information.
tscv = TimeSeriesSplit(n_splits=5)
pred_cv = pd.Series(index=y_train.index, dtype=float, name='forecast')

for fold, (tr_idx, val_idx) in enumerate(tscv.split(x_train), start=1):
    fold_model = clone(model)
    fold_model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
    pred_cv.iloc[val_idx] = fold_model.predict(x_train.iloc[val_idx])
    print(f"  fold {fold}/5 complete")

# Load imbalances data for PnL calculation
imbalances = pd.read_csv('/kaggle/input/ensimag-if-2025/imbalances.csv', 
                          parse_dates=['date'], index_col='date')

# Calculate PnL
def calculate_pnl(row):
    """Calculate profit/loss for a single row"""
    pos_mw = 50
    
    # Too risky - don't trade
    if abs(row['imbalances']) > 1000:
        return 0
    
    if row['forecast'] >= 0:
        # Spread >= 0 => Buy 50MW on Day-Ahead (long position)
        return pos_mw * row['spread']
    else:
        # Spread < 0 => Sell 50MW on Day-Ahead (short position)
        return -pos_mw * row['spread']

# Combine predictions with actual data
# The earliest fold is never in a validation window, so it has no out-of-fold
# prediction; without the dropna those rows fall through calculate_pnl's
# else-branch and get booked as short PnL on a NaN forecast.
concat = pd.concat([pred_cv, imbalances, y_train], axis=1).dropna(subset=['forecast'])
concat['pnl'] = concat.apply(calculate_pnl, axis=1)
concat['pnl_cum'] = concat['pnl'].cumsum()

print(f"\nEstimated Cumulated PnL: €{concat['pnl_cum'].iloc[-1]:,.2f}")

# Visualize cumulated PnL
fig = go.Figure()
fig.add_trace(go.Scatter(x=concat.index, y=concat['pnl_cum'], 
                         mode='lines', name='Cumulated PnL'))
fig.update_layout(title='Walk-Forward CV: Cumulated PnL Over Time',
                  xaxis_title='Date', yaxis_title='Cumulated PnL (€)')
fig.show()
