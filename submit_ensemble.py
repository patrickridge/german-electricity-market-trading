"""Seed-averaged ensemble of the recency-weighted sign classifier.

Six-seed testing showed roughly EUR 2.3M of seed-to-seed variance across two
held-out folds, so the final model averages predicted probabilities over seeds
rather than gambling on one draw.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

import experiment_recency_lags as e
import sign_classifier_model as m

SEEDS = list(range(10))
HALF_LIFE = 730
N_SELECT = 3


def fit_predict(x_tr, y_tr, w_tr, x_va, seed):
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(x_tr, y_tr, sample_weight=w_tr)
    return clf.predict_proba(x_va)[:, 1]


def main():
    train, test = m.load_data()
    tr_rows, te_rows = e.build_extended(train, test)
    spread, imbalance, dates = train["spread"], train["imbalances"], train["date"]

    cols = m.FEATURES + e.EXTRA_FEATURES
    x_train, x_test = tr_rows[cols].to_numpy(), te_rows[cols].to_numpy()
    splits = list(TimeSeriesSplit(n_splits=m.N_SPLITS).split(x_train))
    holdout = splits[N_SELECT:]

    print(f"HELD-OUT folds {N_SELECT + 1}-{len(splits)}, {len(SEEDS)} seeds\n")
    per_seed = np.zeros(len(SEEDS))
    ens_total = perfect_total = 0.0

    for tr, va in holdout:
        s_tr, s_va, i_va = spread.iloc[tr], spread.iloc[va], imbalance.iloc[va]
        y_tr = (s_tr >= 0).astype(int)
        w_tr = e.sample_weights(s_tr, dates.iloc[tr], HALF_LIFE)

        probs = []
        for k, seed in enumerate(SEEDS):
            p = fit_predict(x_train[tr], y_tr, w_tr, x_train[va], seed)
            probs.append(p)
            per_seed[k] += m.pnl(np.where(p >= 0.5, 1, -1), s_va, i_va)

        ens_total += m.pnl(np.where(np.mean(probs, axis=0) >= 0.5, 1, -1), s_va, i_va)
        perfect_total += m.perfect_pnl(s_va, i_va)

    def acc(v):
        return 100 * m.implied_accuracy(v, perfect_total)

    print("individual seeds:")
    for k, seed in enumerate(SEEDS):
        print(f"  seed {seed:<3} EUR {per_seed[k]:>13,.0f}   {acc(per_seed[k]):5.2f}%")

    print(f"\n  mean single seed    EUR {per_seed.mean():>13,.0f}   {acc(per_seed.mean()):5.2f}%")
    print(f"  best single seed    EUR {per_seed.max():>13,.0f}   {acc(per_seed.max()):5.2f}%")
    print(f"  worst single seed   EUR {per_seed.min():>13,.0f}   {acc(per_seed.min()):5.2f}%")
    print(f"  spread best-worst   EUR {per_seed.max() - per_seed.min():>13,.0f}")
    print(f"\n  ENSEMBLE of {len(SEEDS):<2}      EUR {ens_total:>13,.0f}   {acc(ens_total):5.2f}%")
    print(f"  vs mean single seed EUR {ens_total - per_seed.mean():>+13,.0f}")
    print(f"  vs best single seed EUR {ens_total - per_seed.max():>+13,.0f}")

    y, w = (spread >= 0).astype(int), e.sample_weights(spread, dates, HALF_LIFE)
    p_test = np.mean([fit_predict(x_train, y, w, x_test, s) for s in SEEDS], axis=0)
    out = pd.DataFrame({"ID": test["ID"], "forecast": p_test - 0.5})
    out.to_csv("submission_v4.csv", index=False)
    print(f"\nwrote submission_v4.csv ({len(out):,} rows, "
          f"{100 * (out['forecast'] >= 0).mean():.1f}% long)")


if __name__ == "__main__":
    main()
