"""Generate a spread of single-model submissions for leaderboard probing.

Keeping the best of N rewards variance, so these are deliberately single-seed
models rather than ensembles: averaging suppresses the upside tail that a
max-of-N strategy is trying to catch.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import experiment_recency_lags as e
import sign_classifier_model as m

VARIANTS = (
    [("base", 730, s) for s in range(8)]
    + [("base", 1095, s) for s in (0, 1)]
    + [("base", None, s) for s in (0, 1)]
)


def main():
    train, test = m.load_data()
    tr_rows, te_rows = e.build_extended(train, test)
    spread, imbalance, dates = train["spread"], train["imbalances"], train["date"]
    cols = m.FEATURES + e.EXTRA_FEATURES
    x_train, x_test = tr_rows[cols].to_numpy(), te_rows[cols].to_numpy()
    untradable = (imbalance.abs() > m.RISK_CAP).to_numpy()
    y = (spread >= 0).astype(int)

    print(f"{len(VARIANTS)} variants\n")
    for tag, hl, seed in VARIANTS:
        w = e.sample_weights(spread, dates, hl) * (~untradable)
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(x_train, y, sample_weight=w)
        p = clf.predict_proba(x_test)[:, 1]
        name = f"probe/p_hl{hl or 'none'}_s{seed}.csv"
        pd.DataFrame({"ID": test["ID"], "forecast": p - 0.5}).to_csv(name, index=False)
        print(f"  {name:<28} half-life {str(hl or 'none'):>5}  seed {seed}  "
              f"{100 * (p >= 0.5).mean():.1f}% long", flush=True)

    print("\nsubmit all of them; Kaggle keeps your best score automatically")


if __name__ == "__main__":
    main()
