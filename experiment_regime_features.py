"""Regime-relative and cyclical features, tested against the current best model.

Tree ensembles cannot extrapolate past their training range, and the test period
runs structurally greener than any training year. Ratios and rolling z-scores
stay in-range as the underlying levels drift, so they should survive the shift
that raw MW levels do not.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

import experiment_recency_lags as e
import sign_classifier_model as m

SEED = 42
N_SELECT = 3
HALF_LIFE = 730
DAY, MONTH = 96, 2880

REGIME = [
    "solar_rel_30d", "wind_rel_30d", "net_demand_z_30d",
    "renewable_pct_z_30d", "load_z_30d", "net_demand_over_roll30d",
]
CYCLICAL = ["tod_sin", "tod_cos", "doy_sin", "doy_cos"]


def build_all(train, test):
    cols = ["date", "wind", "solar", "load"]
    c = pd.concat([
        train[cols].assign(_src=0, _pos=np.arange(len(train))),
        test[cols].assign(_src=1, _pos=np.arange(len(test))),
    ], ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)

    dt = c["date"]
    c["hour"], c["dayofweek"], c["month"] = dt.dt.hour, dt.dt.dayofweek, dt.dt.month
    c["is_weekend"] = (c["dayofweek"] >= 5).astype(int)
    c["renewable_gen"] = c["wind"] + c["solar"]
    c["renewable_pct"] = c["renewable_gen"] / c["load"]
    c["renewable_pct2"] = c["renewable_pct"] ** 2
    c["renewable_pct3"] = c["renewable_pct"] ** 3
    c["net_demand"] = c["load"] - c["renewable_gen"]
    c["net_demand_ratio"] = c["net_demand"] / c["load"]
    c["wind_ramp"], c["solar_ramp"] = c["wind"].diff(), c["solar"].diff()
    c["load_ramp"], c["net_demand_ramp"] = c["load"].diff(), c["net_demand"].diff()
    c["wind_vol_1h"] = c["wind"].rolling(4).std()
    c["load_vol_1h"] = c["load"].rolling(4).std()
    c["net_demand_vol_1h"] = c["net_demand"].rolling(4).std()
    c["net_demand_lag_1h"] = c["net_demand"].shift(4)
    c["net_demand_lag_24h"] = c["net_demand"].shift(DAY)

    c["net_demand_lag_48h"] = c["net_demand"].shift(2 * DAY)
    c["net_demand_lag_7d"] = c["net_demand"].shift(7 * DAY)
    c["net_demand_roll_24h"] = c["net_demand"].rolling(DAY).mean()
    c["net_demand_roll_7d"] = c["net_demand"].rolling(7 * DAY).mean()
    c["net_demand_dev_24h"] = c["net_demand"] - c["net_demand_roll_24h"]
    c["net_demand_dev_7d"] = c["net_demand"] - c["net_demand_roll_7d"]
    c["net_demand_vol_24h"] = c["net_demand"].rolling(DAY).std()
    c["wind_lag_24h"], c["solar_lag_24h"] = c["wind"].shift(DAY), c["solar"].shift(DAY)
    c["wind_roll_24h"] = c["wind"].rolling(DAY).mean()

    # regime-relative: normalise by the recent local distribution so the feature
    # stays in-range while installed capacity and penetration drift upward
    eps = 1e-6
    solar_cap = c["solar"].rolling(MONTH, min_periods=DAY).max()
    wind_cap = c["wind"].rolling(MONTH, min_periods=DAY).max()
    c["solar_rel_30d"] = c["solar"] / (solar_cap + eps)
    c["wind_rel_30d"] = c["wind"] / (wind_cap + eps)
    for col, name in [("net_demand", "net_demand_z_30d"),
                      ("renewable_pct", "renewable_pct_z_30d"),
                      ("load", "load_z_30d")]:
        mu = c[col].rolling(MONTH, min_periods=DAY).mean()
        sd = c[col].rolling(MONTH, min_periods=DAY).std()
        c[name] = (c[col] - mu) / (sd + eps)
    nd_roll30 = c["net_demand"].rolling(MONTH, min_periods=DAY).mean()
    c["net_demand_over_roll30d"] = c["net_demand"] / (nd_roll30 + eps)

    tod = dt.dt.hour + dt.dt.minute / 60.0
    doy = dt.dt.dayofyear
    c["tod_sin"], c["tod_cos"] = np.sin(2 * np.pi * tod / 24), np.cos(2 * np.pi * tod / 24)
    c["doy_sin"], c["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)

    c = c.replace([np.inf, -np.inf], np.nan)
    return c[c["_src"] == 0].sort_values("_pos"), c[c["_src"] == 1].sort_values("_pos")


def score(x, spread, imbalance, dates, splits, seed=SEED):
    realised = perfect = 0.0
    for tr, va in splits:
        s_tr, s_va, i_va = spread.iloc[tr], spread.iloc[va], imbalance.iloc[va]
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(x[tr], (s_tr >= 0).astype(int),
                sample_weight=e.sample_weights(s_tr, dates.iloc[tr], HALF_LIFE))
        p = clf.predict_proba(x[va])[:, 1]
        realised += m.pnl(np.where(p >= 0.5, 1, -1), s_va, i_va)
        perfect += m.perfect_pnl(s_va, i_va)
    return realised, perfect


def main():
    train, test = m.load_data()
    tr_rows, te_rows = build_all(train, test)
    spread, imbalance, dates = train["spread"], train["imbalances"], train["date"]

    current = m.FEATURES + e.EXTRA_FEATURES
    sets = {
        "current 32": current,
        "+ regime": current + REGIME,
        "+ cyclical": current + CYCLICAL,
        "+ both": current + REGIME + CYCLICAL,
    }
    xs = {k: tr_rows[v].to_numpy() for k, v in sets.items()}
    splits = list(TimeSeriesSplit(n_splits=m.N_SPLITS).split(xs["current 32"]))
    select, holdout = splits[:N_SELECT], splits[N_SELECT:]

    print(f"SELECTION folds 1-{N_SELECT}\n")
    res = {}
    for name, cols in sets.items():
        r, p = score(xs[name], spread, imbalance, dates, select)
        res[name] = r
        print(f"  {name:<12} {len(cols):>3} feats   EUR {r:>13,.0f}   {100 * m.implied_accuracy(r, p):5.2f}%")

    best = max(res, key=res.get)
    print(f"\nbest on selection: {best}")

    print(f"\nHELD-OUT folds {N_SELECT + 1}-{len(splits)}")
    held = {}
    for name in ["current 32", best]:
        r, p = score(xs[name], spread, imbalance, dates, holdout)
        held[name] = r
        print(f"  {name:<12} EUR {r:>13,.0f}   {100 * m.implied_accuracy(r, p):5.2f}%")

    if best != "current 32":
        d = held[best] - held["current 32"]
        print(f"\n  {best} - current = EUR {d:+,.0f}   -> {'PROMISING' if d > 0 else 'rejected'}")


if __name__ == "__main__":
    main()
