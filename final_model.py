"""Final model: tradable-weighted, recency-decayed, seed-averaged sign classifier.

Rows the scorer discards (|imbalance| > 1000) carried 7.1% of the training
weight despite being 0.41% of rows, because weight is proportional to |spread|
and those rows have the largest spreads. Zeroing them aligns training with what
is actually scored.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

import experiment_recency_lags as e
import experiment_regime_features as g
import sign_classifier_model as m

HALF_LIFE = 730
COMPARE_SEEDS = [0, 1, 2, 3, 4, 42]
ENSEMBLE_SEEDS = list(range(10))


def weights(spread, dates, untradable):
    return e.sample_weights(spread, dates, HALF_LIFE) * (~untradable).to_numpy()


def main():
    train, test = m.load_data()
    tr_rows, te_rows = g.build_all(train, test)
    spread, imbalance, dates = train["spread"], train["imbalances"], train["date"]
    untradable = imbalance.abs() > m.RISK_CAP

    base = m.FEATURES + e.EXTRA_FEATURES
    sets = {"tradable": base, "tradable + regime": base + g.REGIME}
    xs = {k: tr_rows[v].to_numpy() for k, v in sets.items()}
    splits = list(TimeSeriesSplit(n_splits=m.N_SPLITS).split(xs["tradable"]))
    holdout = splits[3:]

    print(f"HELD-OUT folds 4-5, {len(COMPARE_SEEDS)} matched seeds\n")
    totals = {k: [] for k in sets}
    for seed in COMPARE_SEEDS:
        for name in sets:
            realised = perfect = 0.0
            for tr, va in holdout:
                s_tr, s_va, i_va = spread.iloc[tr], spread.iloc[va], imbalance.iloc[va]
                clf = HistGradientBoostingClassifier(random_state=seed)
                clf.fit(xs[name][tr], (s_tr >= 0).astype(int),
                        sample_weight=weights(s_tr, dates.iloc[tr], untradable.iloc[tr]))
                p = clf.predict_proba(xs[name][va])[:, 1]
                realised += m.pnl(np.where(p >= 0.5, 1, -1), s_va, i_va)
                perfect += m.perfect_pnl(s_va, i_va)
            totals[name].append(realised)
        print(f"  seed {seed:<3} " + "  ".join(
            f"{n}: EUR {totals[n][-1]:>12,.0f}" for n in sets))

    means = {k: float(np.mean(v)) for k, v in totals.items()}
    print()
    for name, v in means.items():
        print(f"  {name:<20} mean EUR {v:>13,.0f}")
    winner = max(means, key=means.get)
    print(f"\n  winner: {winner}  (+EUR {means[winner] - min(means.values()):,.0f})")

    cols = sets[winner]
    x_train, x_test = tr_rows[cols].to_numpy(), te_rows[cols].to_numpy()
    y = (spread >= 0).astype(int)
    w = weights(spread, dates, untradable)

    probs = []
    for seed in ENSEMBLE_SEEDS:
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(x_train, y, sample_weight=w)
        probs.append(clf.predict_proba(x_test)[:, 1])

    out = pd.DataFrame({"ID": test["ID"], "forecast": np.mean(probs, axis=0) - 0.5})
    out.to_csv("submission_v5.csv", index=False)
    print(f"\nwrote submission_v5.csv  ({len(out):,} rows, {len(cols)} features, "
          f"{len(ENSEMBLE_SEEDS)} seeds, {100 * (out['forecast'] >= 0).mean():.1f}% long)")


if __name__ == "__main__":
    main()
