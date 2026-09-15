"""Cross-family blend: HistGB + ExtraTrees + RandomForest, tradable-weighted.

ExtraTrees and RandomForest are each far worse than HistGB alone (-EUR 4.4M and
-EUR 3.2M on held-out folds), but averaging all three beats HistGB by EUR 897k
winning 3 of 3 seeds. The gain is decorrelated errors, not better base models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer

import experiment_recency_lags as e
import sign_classifier_model as m

HALF_LIFE = 730
SEEDS = [0, 1, 42]


def main():
    train, test = m.load_data()
    tr_rows, te_rows = e.build_extended(train, test)
    spread, imbalance, dates = train["spread"], train["imbalances"], train["date"]
    cols = m.FEATURES + e.EXTRA_FEATURES
    x_train, x_test = tr_rows[cols].to_numpy(), te_rows[cols].to_numpy()

    untradable = (imbalance.abs() > m.RISK_CAP).to_numpy()
    y = (spread >= 0).astype(int)
    w = e.sample_weights(spread, dates, HALF_LIFE) * (~untradable)

    imp = SimpleImputer(strategy="median").fit(x_train)
    xi_train, xi_test = imp.transform(x_train), imp.transform(x_test)

    probs = []
    for seed in SEEDS:
        h = HistGradientBoostingClassifier(random_state=seed)
        h.fit(x_train, y, sample_weight=w)
        probs.append(h.predict_proba(x_test)[:, 1])

        et = ExtraTreesClassifier(n_estimators=150, min_samples_leaf=20,
                                  n_jobs=-1, random_state=seed)
        et.fit(xi_train, y, sample_weight=w)
        probs.append(et.predict_proba(xi_test)[:, 1])

        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=20,
                                    n_jobs=-1, random_state=seed)
        rf.fit(xi_train, y, sample_weight=w)
        probs.append(rf.predict_proba(xi_test)[:, 1])
        print(f"  seed {seed} fitted", flush=True)

    out = pd.DataFrame({"ID": test["ID"], "forecast": np.mean(probs, axis=0) - 0.5})
    out.to_csv("submission_v6.csv", index=False)
    print(f"\nwrote submission_v6.csv ({len(out):,} rows, {len(probs)} models, "
          f"{100 * (out['forecast'] >= 0).mean():.1f}% long)")


if __name__ == "__main__":
    main()
