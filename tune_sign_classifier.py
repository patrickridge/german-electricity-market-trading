"""Nested hyperparameter search for the sign classifier.

Configs are ranked on early walk-forward folds; the winner is then re-scored on
later folds that never influenced the choice, so the reported gain is not
selection-biased.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

import sign_classifier_model as m

N_CONFIGS = 24
N_SELECT_FOLDS = 3
SEED = 42

GRID = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [20, 50, 200, 500],
    "max_iter": [100, 200, 400],
    "l2_regularization": [0.0, 0.1, 1.0],
}
DEFAULT = {
    "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
    "max_iter": 100, "l2_regularization": 0.0,
}
INT_KEYS = ("max_leaf_nodes", "min_samples_leaf", "max_iter")


def sample_configs(n, rng):
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in GRID.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def score_config(x, spread, imbalance, splits, params):
    clean = {k: (int(v) if k in INT_KEYS else float(v)) for k, v in params.items()}
    realised = perfect = 0.0
    for tr, va in splits:
        s_tr, s_va, i_va = spread.iloc[tr], spread.iloc[va], imbalance.iloc[va]
        clf = HistGradientBoostingClassifier(
            random_state=SEED, early_stopping=False, **clean
        )
        clf.fit(x[tr], (s_tr >= 0).astype(int), sample_weight=s_tr.abs().to_numpy())
        p = clf.predict_proba(x[va])[:, 1]
        realised += m.pnl(np.where(p >= 0.5, 1, -1), s_va, i_va)
        perfect += m.perfect_pnl(s_va, i_va)
    return realised, perfect


def main():
    train, test = m.load_data()
    x_train, x_test = m.build_features(train, test)
    spread, imbalance = train["spread"], train["imbalances"]

    splits = list(TimeSeriesSplit(n_splits=m.N_SPLITS).split(x_train))
    select, holdout = splits[:N_SELECT_FOLDS], splits[N_SELECT_FOLDS:]

    configs = sample_configs(N_CONFIGS, np.random.default_rng(SEED))
    print(f"scoring {len(configs)} configs on folds 1-{N_SELECT_FOLDS}\n")

    rows = []
    for i, cfg in enumerate(configs, 1):
        r, p = score_config(x_train, spread, imbalance, select, cfg)
        acc = 100 * m.implied_accuracy(r, p)
        rows.append({**cfg, "select_pnl": r, "select_acc": acc})
        print(f"  [{i:>2}/{len(configs)}] EUR {r:>13,.0f}  {acc:5.2f}%  lr={cfg['learning_rate']:<5} "
              f"leaves={cfg['max_leaf_nodes']:<4} minleaf={cfg['min_samples_leaf']:<4} "
              f"iter={cfg['max_iter']:<4} l2={cfg['l2_regularization']}")

    res = pd.DataFrame(rows).sort_values("select_pnl", ascending=False).reset_index(drop=True)
    best = {k: res.loc[0, k] for k in GRID}
    print(f"\nbest on selection folds: {best}")

    print(f"\nHELD-OUT folds {N_SELECT_FOLDS + 1}-{len(splits)} (never used for selection)")
    scores = {}
    for label, cfg in [("default", DEFAULT), ("tuned", best)]:
        r, p = score_config(x_train, spread, imbalance, holdout, cfg)
        scores[label] = r
        print(f"  {label:<8} EUR {r:>13,.0f}   implied acc {100 * m.implied_accuracy(r, p):.2f}%")

    delta = scores["tuned"] - scores["default"]
    print(f"\n  tuned - default = EUR {delta:+,.0f}  ({'keep tuned' if delta > 0 else 'keep default'})")

    clean = {k: (int(v) if k in INT_KEYS else float(v)) for k, v in best.items()}
    clf = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **clean)
    clf.fit(x_train, (spread >= 0).astype(int), sample_weight=spread.abs().to_numpy())
    out = pd.DataFrame({
        "ID": test["ID"],
        "forecast": clf.predict_proba(x_test)[:, 1] - 0.5,
    })
    out.to_csv("submission_sign_classifier_tuned.csv", index=False)
    print(f"\nwrote submission_sign_classifier_tuned.csv "
          f"({len(out):,} rows, {100 * (out['forecast'] >= 0).mean():.1f}% long)")


if __name__ == "__main__":
    main()
