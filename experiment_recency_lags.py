"""Walk-forward tests for recency weighting and an expanded lag set.

Variants are ranked on early folds and the winner re-scored on later folds that
never influenced the choice — the hyperparameter search showed that a flat
search over all folds reports gains which do not generalise.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

import sign_classifier_model as m

SEED = 42
N_SELECT = 3
HALF_LIVES = [None, 180, 365, 730]

EXTRA_FEATURES = [
    "net_demand_lag_48h", "net_demand_lag_7d",
    "net_demand_roll_24h", "net_demand_roll_7d",
    "net_demand_dev_24h", "net_demand_dev_7d",
    "net_demand_vol_24h", "wind_lag_24h", "solar_lag_24h", "wind_roll_24h",
]


def build_extended(train, test):
    cols = ["date", "wind", "solar", "load"]
    c = pd.concat([
        train[cols].assign(_src=0, _pos=np.arange(len(train))),
        test[cols].assign(_src=1, _pos=np.arange(len(test))),
    ], ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)

    dt = c["date"]
    c["hour"] = dt.dt.hour
    c["dayofweek"] = dt.dt.dayofweek
    c["month"] = dt.dt.month
    c["is_weekend"] = (c["dayofweek"] >= 5).astype(int)
    c["renewable_gen"] = c["wind"] + c["solar"]
    c["renewable_pct"] = c["renewable_gen"] / c["load"]
    c["renewable_pct2"] = c["renewable_pct"] ** 2
    c["renewable_pct3"] = c["renewable_pct"] ** 3
    c["net_demand"] = c["load"] - c["renewable_gen"]
    c["net_demand_ratio"] = c["net_demand"] / c["load"]
    c["wind_ramp"] = c["wind"].diff()
    c["solar_ramp"] = c["solar"].diff()
    c["load_ramp"] = c["load"].diff()
    c["net_demand_ramp"] = c["net_demand"].diff()
    c["wind_vol_1h"] = c["wind"].rolling(4).std()
    c["load_vol_1h"] = c["load"].rolling(4).std()
    c["net_demand_vol_1h"] = c["net_demand"].rolling(4).std()
    c["net_demand_lag_1h"] = c["net_demand"].shift(4)
    c["net_demand_lag_24h"] = c["net_demand"].shift(96)

    c["net_demand_lag_48h"] = c["net_demand"].shift(192)
    c["net_demand_lag_7d"] = c["net_demand"].shift(672)
    c["net_demand_roll_24h"] = c["net_demand"].rolling(96).mean()
    c["net_demand_roll_7d"] = c["net_demand"].rolling(672).mean()
    c["net_demand_dev_24h"] = c["net_demand"] - c["net_demand_roll_24h"]
    c["net_demand_dev_7d"] = c["net_demand"] - c["net_demand_roll_7d"]
    c["net_demand_vol_24h"] = c["net_demand"].rolling(96).std()
    c["wind_lag_24h"] = c["wind"].shift(96)
    c["solar_lag_24h"] = c["solar"].shift(96)
    c["wind_roll_24h"] = c["wind"].rolling(96).mean()

    c = c.replace([np.inf, -np.inf], np.nan)
    tr = c[c["_src"] == 0].sort_values("_pos")
    te = c[c["_src"] == 1].sort_values("_pos")
    assert len(tr) == len(train) and len(te) == len(test)
    return tr, te


def sample_weights(spread, dates, half_life):
    w = spread.abs().to_numpy()
    if half_life is not None:
        age = (dates.max() - dates).dt.days.to_numpy()
        w = w * (0.5 ** (age / half_life))
    return w


def score(x, spread, imbalance, dates, splits, half_life):
    realised = perfect = 0.0
    for tr, va in splits:
        s_tr, s_va, i_va = spread.iloc[tr], spread.iloc[va], imbalance.iloc[va]
        clf = HistGradientBoostingClassifier(random_state=SEED)
        clf.fit(x[tr], (s_tr >= 0).astype(int),
                sample_weight=sample_weights(s_tr, dates.iloc[tr], half_life))
        p = clf.predict_proba(x[va])[:, 1]
        realised += m.pnl(np.where(p >= 0.5, 1, -1), s_va, i_va)
        perfect += m.perfect_pnl(s_va, i_va)
    return realised, perfect


def main():
    train, test = m.load_data()
    tr_rows, te_rows = build_extended(train, test)
    spread, imbalance, dates = train["spread"], train["imbalances"], train["date"]

    sets = {
        "current 22": m.FEATURES,
        "expanded 32": m.FEATURES + EXTRA_FEATURES,
    }
    x_by_set = {k: tr_rows[v].to_numpy() for k, v in sets.items()}
    splits = list(TimeSeriesSplit(n_splits=m.N_SPLITS).split(x_by_set["current 22"]))
    select, holdout = splits[:N_SELECT], splits[N_SELECT:]

    print(f"SELECTION folds 1-{N_SELECT}\n")
    print(f"{'features':<14}{'half-life':>12}{'PnL':>18}{'acc':>9}")
    results = {}
    for sname, cols in sets.items():
        for hl in HALF_LIVES:
            r, p = score(x_by_set[sname], spread, imbalance, dates, select, hl)
            acc = 100 * m.implied_accuracy(r, p)
            results[(sname, hl)] = r
            print(f"{sname:<14}{str(hl) + 'd' if hl else 'none':>12}"
                  f"{'EUR ' + format(r, ',.0f'):>18}{acc:>8.2f}%")

    best = max(results, key=results.get)
    base = ("current 22", None)
    print(f"\nbest on selection folds: {best[0]}, half-life {best[1] or 'none'}")

    print(f"\nHELD-OUT folds {N_SELECT + 1}-{len(splits)} (never used for selection)")
    held = {}
    for label, key in [("baseline (submitted)", base), ("best variant", best)]:
        r, p = score(x_by_set[key[0]], spread, imbalance, dates, holdout, key[1])
        held[label] = r
        print(f"  {label:<22}{'EUR ' + format(r, ',.0f'):>18}"
              f"   implied acc {100 * m.implied_accuracy(r, p):.2f}%")

    delta = held["best variant"] - held["baseline (submitted)"]
    verdict = "SUBMIT IT" if delta > 0 else "keep the current model"
    print(f"\n  best - baseline = EUR {delta:+,.0f}   -> {verdict}")

    if delta > 0:
        clf = HistGradientBoostingClassifier(random_state=SEED)
        clf.fit(x_by_set[best[0]], (spread >= 0).astype(int),
                sample_weight=sample_weights(spread, dates, best[1]))
        x_test = te_rows[sets[best[0]]].to_numpy()
        out = pd.DataFrame({"ID": test["ID"],
                            "forecast": clf.predict_proba(x_test)[:, 1] - 0.5})
        out.to_csv("submission_v3.csv", index=False)
        print(f"  wrote submission_v3.csv ({len(out):,} rows, "
              f"{100 * (out['forecast'] >= 0).mean():.1f}% long)")


if __name__ == "__main__":
    main()
