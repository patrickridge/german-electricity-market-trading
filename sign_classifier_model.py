"""Sign-classifier model for the Ensimag IF spread competition.

The scorer reads only the sign of the forecast, so this trains that decision
directly: classification of sign(spread) weighted by |spread|, which puts the
0.5 probability contour exactly at E[spread] = 0, the PnL-optimal boundary.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = "data"
POS_MW = 50
RISK_CAP = 1000
N_SPLITS = 5
SEED = 42

BASE_FEATURES = [
    "hour", "dayofweek", "month", "is_weekend",
    "wind", "solar", "load",
    "renewable_gen", "renewable_pct", "renewable_pct2", "renewable_pct3",
    "net_demand", "net_demand_ratio",
]
DYNAMIC_FEATURES = [
    "wind_ramp", "solar_ramp", "load_ramp", "net_demand_ramp",
    "wind_vol_1h", "load_vol_1h", "net_demand_vol_1h",
    "net_demand_lag_1h", "net_demand_lag_24h",
]
FEATURES = BASE_FEATURES + DYNAMIC_FEATURES


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["date"])
    imbalances = pd.read_csv(f"{DATA_DIR}/imbalances.csv", parse_dates=["date"])
    train = train.merge(imbalances, on="date", how="left")
    return train, test


def build_features(train, test):
    cols = ["date", "wind", "solar", "load"]
    combined = pd.concat(
        [
            train[cols].assign(_src=0, _pos=np.arange(len(train))),
            test[cols].assign(_src=1, _pos=np.arange(len(test))),
        ],
        ignore_index=True,
    ).sort_values("date", kind="mergesort").reset_index(drop=True)

    dt = combined["date"]
    combined["hour"] = dt.dt.hour
    combined["dayofweek"] = dt.dt.dayofweek
    combined["month"] = dt.dt.month
    combined["is_weekend"] = (combined["dayofweek"] >= 5).astype(int)

    combined["renewable_gen"] = combined["wind"] + combined["solar"]
    combined["renewable_pct"] = combined["renewable_gen"] / combined["load"]
    combined["renewable_pct2"] = combined["renewable_pct"] ** 2
    combined["renewable_pct3"] = combined["renewable_pct"] ** 3
    combined["net_demand"] = combined["load"] - combined["renewable_gen"]
    combined["net_demand_ratio"] = combined["net_demand"] / combined["load"]

    combined["wind_ramp"] = combined["wind"].diff()
    combined["solar_ramp"] = combined["solar"].diff()
    combined["load_ramp"] = combined["load"].diff()
    combined["net_demand_ramp"] = combined["net_demand"].diff()
    combined["wind_vol_1h"] = combined["wind"].rolling(4).std()
    combined["load_vol_1h"] = combined["load"].rolling(4).std()
    combined["net_demand_vol_1h"] = combined["net_demand"].rolling(4).std()
    combined["net_demand_lag_1h"] = combined["net_demand"].shift(4)
    combined["net_demand_lag_24h"] = combined["net_demand"].shift(96)

    combined = combined.replace([np.inf, -np.inf], np.nan)

    tr_rows = combined[combined["_src"] == 0].sort_values("_pos")
    te_rows = combined[combined["_src"] == 1].sort_values("_pos")
    assert len(tr_rows) == len(train) and len(te_rows) == len(test)
    return tr_rows[FEATURES].to_numpy(), te_rows[FEATURES].to_numpy()


def pnl(decision, spread, imbalance):
    tradable = (imbalance.abs() <= RISK_CAP).to_numpy()
    return float(np.sum(POS_MW * decision * spread.to_numpy() * tradable))


def perfect_pnl(spread, imbalance):
    tradable = (imbalance.abs() <= RISK_CAP).to_numpy()
    return float(np.sum(POS_MW * spread.abs().to_numpy() * tradable))


def implied_accuracy(realised, perfect):
    return 0.5 * (1.0 + realised / perfect)


def tune_threshold(proba, spread, imbalance):
    grid = np.linspace(0.30, 0.70, 81)
    scores = [pnl(np.where(proba >= t, 1, -1), spread, imbalance) for t in grid]
    return float(grid[int(np.argmax(scores))])


def walk_forward(x, spread, imbalance):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows, oof_proba, oof_idx = [], [], []

    for fold, (tr, va) in enumerate(tscv.split(x), start=1):
        s_tr, s_va, i_va = spread.iloc[tr], spread.iloc[va], imbalance.iloc[va]

        reg = HistGradientBoostingRegressor(random_state=SEED)
        reg.fit(x[tr], s_tr)
        d_reg = np.where(reg.predict(x[va]) >= 0, 1, -1)

        clf = HistGradientBoostingClassifier(random_state=SEED)
        clf.fit(x[tr], (s_tr >= 0).astype(int), sample_weight=s_tr.abs().to_numpy())
        p_va = clf.predict_proba(x[va])[:, 1]

        # threshold is fitted only on earlier folds, never on the fold being scored
        if oof_proba:
            prev = np.concatenate(oof_idx)
            thr = tune_threshold(
                np.concatenate(oof_proba), spread.iloc[prev], imbalance.iloc[prev]
            )
        else:
            thr = 0.5

        rows.append({
            "fold": fold,
            "n_val": len(va),
            "perfect": perfect_pnl(s_va, i_va),
            "always_long": pnl(np.ones(len(va)), s_va, i_va),
            "always_short": pnl(-np.ones(len(va)), s_va, i_va),
            "mse_regressor": pnl(d_reg, s_va, i_va),
            "clf_050": pnl(np.where(p_va >= 0.5, 1, -1), s_va, i_va),
            "clf_tuned": pnl(np.where(p_va >= thr, 1, -1), s_va, i_va),
            "thr_used": thr,
        })
        oof_proba.append(p_va)
        oof_idx.append(va)

    return pd.DataFrame(rows), np.concatenate(oof_proba), np.concatenate(oof_idx)


def main():
    train, test = load_data()
    x_train, x_test = build_features(train, test)
    spread, imbalance = train["spread"], train["imbalances"]

    print(f"train {x_train.shape}  test {x_test.shape}  features {len(FEATURES)}\n")

    folds, oof_proba, oof_idx = walk_forward(x_train, spread, imbalance)
    pd.set_option("display.width", 200)
    print("PER-FOLD PnL (walk-forward, EUR)")
    shown = folds.drop(columns=["perfect"]).copy()
    shown["thr_used"] = shown["thr_used"].map("{:.3f}".format)
    print(shown.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    strategies = ["always_long", "always_short", "mse_regressor", "clf_050", "clf_tuned"]
    total_perfect = folds["perfect"].sum()
    print(f"\nTOTALS across all folds (perfect foresight = EUR {total_perfect:,.0f})")
    for s in strategies:
        v = folds[s].sum()
        print(f"  {s:<16} EUR {v:>14,.0f}   {100 * v / total_perfect:>5.1f}% of max"
              f"   implied acc {100 * implied_accuracy(v, total_perfect):.2f}%")

    thr_final = tune_threshold(oof_proba, spread.iloc[oof_idx], imbalance.iloc[oof_idx])
    print(f"\nfinal decision threshold (all out-of-fold preds): {thr_final:.3f}")

    clf = HistGradientBoostingClassifier(random_state=SEED)
    clf.fit(x_train, (spread >= 0).astype(int), sample_weight=spread.abs().to_numpy())
    p_test = clf.predict_proba(x_test)[:, 1]

    out = pd.DataFrame({"ID": test["ID"], "forecast": p_test - thr_final})
    out.to_csv("submission_sign_classifier.csv", index=False)
    long_share = 100 * (out["forecast"] >= 0).mean()
    print(f"wrote submission_sign_classifier.csv  ({len(out):,} rows, {long_share:.1f}% long)")


if __name__ == "__main__":
    main()
