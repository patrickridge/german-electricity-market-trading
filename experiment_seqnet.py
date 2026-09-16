"""Dilated temporal CNN over the raw 15-minute series.

The tree models only ever saw hand-picked lags. This gives a model the raw
24-hour window of wind/solar/load/net-demand alongside the same tabular
features, so any gain is attributable to sequence structure the lags miss.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit

import experiment_recency_lags as e
import sign_classifier_model as m

WINDOW = 96          # 24 hours of 15-minute steps
CHANNELS = ["wind", "solar", "load", "net_demand"]
HALF_LIFE = 730
EPOCHS = 8
BATCH = 512
torch.set_num_threads(10)


class SeqNet(nn.Module):
    def __init__(self, n_ch, n_static):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + n_static, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, seq, static):
        h = self.conv(seq)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(torch.cat([pooled, static], dim=1)).squeeze(1)


def train_eval(seq_all, stat_all, y, w, tr, va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    mu, sd = stat_all[tr].mean(0), stat_all[tr].std(0) + 1e-6
    stat = np.nan_to_num((stat_all - mu) / sd).astype(np.float32)
    cmu = seq_all[tr].mean(axis=(0, 2), keepdims=True)
    csd = seq_all[tr].std(axis=(0, 2), keepdims=True) + 1e-6

    model = SeqNet(seq_all.shape[1], stat.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    wt = w[tr] / w[tr].mean()
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(order), BATCH):
            idx = tr[order[i:i + BATCH]]
            bw = torch.from_numpy(wt[order[i:i + BATCH]].astype(np.float32))
            s = torch.from_numpy(((seq_all[idx] - cmu) / csd).astype(np.float32))
            st = torch.from_numpy(stat[idx])
            yy = torch.from_numpy(y[idx].astype(np.float32))
            opt.zero_grad()
            loss = (lossf(model(s, st), yy) * bw).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"      epoch {ep + 1}/{EPOCHS}  loss {tot / len(order):.4f}", flush=True)

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(va), 2048):
            idx = va[i:i + 2048]
            s = torch.from_numpy(((seq_all[idx] - cmu) / csd).astype(np.float32))
            st = torch.from_numpy(stat[idx])
            out.append(torch.sigmoid(model(s, st)).numpy())
    return np.concatenate(out)


def main():
    train, test = m.load_data()
    tr_rows, _ = e.build_extended(train, test)
    spread, imbalance, dates = train["spread"], train["imbalances"], train["date"]

    raw = tr_rows[CHANNELS].to_numpy(np.float32)
    raw = np.nan_to_num(raw)
    windows = np.lib.stride_tricks.sliding_window_view(raw, WINDOW, axis=0)  # (N-95, C, T)
    stat_all = tr_rows[m.FEATURES + e.EXTRA_FEATURES].to_numpy(np.float32)

    y = (spread >= 0).astype(np.float32).to_numpy()
    untradable = (imbalance.abs() > m.RISK_CAP).to_numpy()
    w = (e.sample_weights(spread, dates, HALF_LIFE) * (~untradable)).astype(np.float64)

    offset = WINDOW - 1                     # row j uses window index j - offset
    n = len(windows)
    splits = list(TimeSeriesSplit(n_splits=m.N_SPLITS).split(np.arange(len(tr_rows))))
    print(f"windows {windows.shape}, static {stat_all.shape}\n")

    realised = perfect = 0.0
    for fi, (tr, va) in enumerate(splits[3:], start=4):
        tr = tr[tr >= offset] - offset
        va_rows = va[va >= offset]
        va = va_rows - offset
        print(f"  fold {fi}: train {len(tr):,}  val {len(va):,}", flush=True)
        p = train_eval(windows, stat_all[offset:], y[offset:], w[offset:], tr, va, seed=0)
        s_va, i_va = spread.iloc[va_rows], imbalance.iloc[va_rows]
        fold_pnl = m.pnl(np.where(p >= 0.5, 1, -1), s_va, i_va)
        realised += fold_pnl
        perfect += m.perfect_pnl(s_va, i_va)
        print(f"    fold {fi} PnL EUR {fold_pnl:,.0f}\n", flush=True)

    print(f"SEQNET  EUR {realised:,.0f}   implied acc {100 * m.implied_accuracy(realised, perfect):.2f}%")
    print(f"tree baseline on the same folds was EUR 49,958,345 (59.37%)")
    print(f"delta EUR {realised - 49_958_345:+,.0f}")


if __name__ == "__main__":
    main()
