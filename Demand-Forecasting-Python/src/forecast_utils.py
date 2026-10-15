"""Shared helpers for the demand-forecasting notebooks: metrics, backtest folds, feature builder.

Author: Anurag Moyal
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- metrics
def metrics(y, yhat) -> dict:
    """Standard point-forecast error metrics (y and yhat are arrays of equal shape)."""
    y, yhat = np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()
    err = yhat - y
    nz = y > 0
    return dict(
        MAE=np.mean(np.abs(err)),
        RMSE=np.sqrt(np.mean(err**2)),
        MAPE=np.mean(np.abs(err[nz]) / y[nz]) * 100,
        sMAPE=np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(yhat) + 1e-9)) * 100,
        WAPE=np.sum(np.abs(err)) / np.sum(np.abs(y)) * 100,
        Bias=np.mean(err) / np.mean(y) * 100,
    )


def mase(y, yhat, y_train, m=7) -> float:
    """Mean absolute scaled error against the in-sample seasonal-naive forecast (scale-free)."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))) / scale)


# ----------------------------------------------------------------------------- backtesting
def rolling_origins(dates: pd.DatetimeIndex, horizon: int, n_folds: int, step: int | None = None):
    """Return n_folds expanding-window folds, ordered oldest -> newest, ending at the last date.

    Each fold = (train_end_idx, test_start_idx, test_end_idx) as positional indices into `dates`.
    Folds do not overlap when step == horizon (the default).
    """
    step = step or horizon
    last = len(dates) - 1
    folds = []
    for k in range(n_folds):
        test_end = last - k * step
        test_start = test_end - horizon + 1
        folds.append((test_start - 1, test_start, test_end))
    return folds[::-1]


# ----------------------------------------------------------------------------- direct multi-horizon features
FEATURES = ["h", "dow", "month", "dom", "doy", "m7", "m28", "m91", "m365", "dow_mean8", "yoy", "yoy_smooth", "growth",
            "trend28_91", "store", "item"]


def _csum(S):
    return np.vstack([np.zeros((1, S.shape[1])), np.cumsum(S, axis=0)])


def build_rows(S: np.ndarray, dates: pd.DatetimeIndex, origins, horizon: int, cutoff: int | None = None,
               store=None, item=None, with_target: bool = True) -> pd.DataFrame:
    """Build one row per (origin, horizon-step, series) using ONLY information available at the origin.

    S       : array (T, N) of daily sales, rows aligned with `dates`.
    origins : positional indices of the last observed day for each forecast origin.
    cutoff  : if given, rows whose target day is after `cutoff` are dropped (prevents leakage when training).
    """
    T, N = S.shape
    C = _csum(S)
    mean_win = lambda o, w: (C[o + 1] - C[o + 1 - w]) / w          # mean of S[o-w+1 .. o]
    store = np.zeros(N, int) if store is None else np.asarray(store)
    item = np.zeros(N, int) if item is None else np.asarray(item)
    out = []
    for o in origins:
        if o < 365 + 7:
            continue
        m7, m28, m91, m365 = (mean_win(o, w) for w in (7, 28, 91, 365))
        m91_ly = (C[o - 364 + 1] - C[o - 364 + 1 - 91]) / 91           # same 91-day window one year earlier
        growth = m91 / np.maximum(m91_ly, 1e-6)
        trend = m28 / np.maximum(m91, 1e-6)
        for h in range(1, horizon + 1):
            t = o + h
            if t >= T and with_target:
                break
            if cutoff is not None and t > cutoff:
                break
            td = dates[t] if t < T else dates[-1] + pd.Timedelta(days=t - T + 1)
            ks = np.arange(int(np.ceil(h / 7)), int(np.ceil(h / 7)) + 8)
            dow_mean = S[t - 7 * ks].mean(axis=0)                       # same weekday, last 8 observed weeks
            ly = S[t - 364]
            ly_s = S[t - 364 - 3: t - 364 + 4].mean(axis=0)
            row = dict(h=np.full(N, h), dow=np.full(N, td.dayofweek), month=np.full(N, td.month), dom=np.full(N, td.day),
                       doy=np.full(N, td.dayofyear), m7=m7, m28=m28, m91=m91, m365=m365, dow_mean8=dow_mean, yoy=ly,
                       yoy_smooth=ly_s, growth=growth, trend28_91=trend, store=store, item=item,
                       origin=np.full(N, o), t=np.full(N, t), series=np.arange(N))
            if with_target:
                row["y"] = S[t]
            out.append(pd.DataFrame(row))
    return pd.concat(out, ignore_index=True)
