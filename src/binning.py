# -*- coding: utf-8 -*-
"""Binning and WOE.

Cut points are chosen on train and WOE is computed over them. The result is a
plain dict, so inference needs nothing but a lookup over the cut points.
"""
import numpy as np
import pandas as pd

from . import config as cfg

MISSING = 'MISSING'


def _woe_table(labels, y, smooth=cfg.SMOOTH):
    table = pd.crosstab(pd.Series(labels), pd.Series(np.asarray(y)))
    for cls in (0, 1):
        if cls not in table.columns:
            table[cls] = 0

    k = len(table)
    good = (table[0] + smooth) / ((y == 0).sum() + k * smooth)
    bad = (table[1] + smooth) / ((y == 1).sum() + k * smooth)
    woe = np.log(good / bad)
    iv = float(((good - bad) * woe).sum())
    return woe, iv


def _quantile_edges(x, n_bins):
    q = np.nanquantile(x, np.linspace(0, 1, n_bins + 1))
    return np.unique(q)[1:-1]


def _prebin_edges(x, n_bins):
    """Starting cut points before merging.

    Quantiles cannot reach the sparse tail of a skewed discrete feature: the
    95th percentile of the active-credit count is 5 while the tail runs to 32,
    so every value above 5 would end up in one bin. For features with few
    distinct values the value grid itself is used instead.
    """
    values = np.unique(x[~np.isnan(x)])
    if len(values) <= cfg.DISCRETE_MAX_LEVELS:
        return (values[:-1] + values[1:]) / 2
    return _quantile_edges(x, n_bins)


def _bin_stats(x, y, edges):
    idx = np.searchsorted(edges, x, side='right')
    n = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    bad = np.bincount(idx, weights=y, minlength=len(edges) + 1)
    rate = np.divide(bad, n, out=np.full(n.shape, np.nan), where=n > 0)
    return n, rate


def _monotone_edges(x, y, direction, n_prebins=None, min_size=None):
    """Quantile bins merged until the default rate becomes monotone.

    direction: 'ascending'  higher value, higher default rate
               'descending' lower
    """
    n_prebins = n_prebins or cfg.N_PREBINS
    min_size = min_size or cfg.MIN_BIN_SIZE

    ok = ~np.isnan(x)
    x, y = x[ok], np.asarray(y)[ok]
    edges = _prebin_edges(x, n_prebins)

    while len(edges):
        n, rate = _bin_stats(x, y, edges)

        small = np.flatnonzero(n < min_size * len(x))
        if len(small):
            i = small[0]
            edges = np.delete(edges, 0 if i == 0 else i - 1)
            continue

        step = np.diff(rate)
        violated = step < 0 if direction == 'ascending' else step > 0
        if not violated.any():
            break

        edges = np.delete(edges, int(np.argmin(np.where(violated, np.abs(step), np.inf))))

    return [float(e) for e in edges]


def _fit_numeric(x, y, method, name=None):
    if method == 'monotonic':
        direction = cfg.MONOTONIC_DIRECTION.get(name)
        if direction is None:
            edges = [float(e) for e in _quantile_edges(x, cfg.N_BINS)]
        else:
            edges = _monotone_edges(x, y, direction)
    else:
        edges = [float(e) for e in _quantile_edges(x, cfg.N_BINS)]

    idx = np.searchsorted(edges, np.asarray(x, dtype=float), side='right')
    labels = np.where(pd.isna(x), MISSING, idx.astype(str))

    woe, iv = _woe_table(labels, y)
    woe = woe.to_dict()

    return {
        'kind': 'numeric',
        'edges': edges,
        'woe': [float(woe.get(str(i), 0.0)) for i in range(len(edges) + 1)],
        'missing_woe': float(woe.get(MISSING, 0.0)),
        'iv': iv,
    }


def _fit_categorical(x, y):
    labels = pd.Series(x).astype('object')
    labels = labels.where(labels.notna(), MISSING).astype(str)

    woe, iv = _woe_table(labels.to_numpy(), y)
    woe = {str(k): float(v) for k, v in woe.to_dict().items()}

    return {
        'kind': 'categorical',
        'woe': {k: v for k, v in woe.items() if k != MISSING},
        'missing_woe': float(woe.get(MISSING, 0.0)),
        'unseen_woe': 0.0,
        'iv': iv,
    }


def fit(X, y, features, categorical, method=None):
    method = method or cfg.BINNING
    spec = {}
    for col in features:
        if col in categorical:
            spec[col] = _fit_categorical(X[col].to_numpy(), y)
        else:
            spec[col] = _fit_numeric(X[col].to_numpy(dtype=float), y, method, col)
    return spec


def transform_value(rule, value):
    """A single value -> WOE."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return rule['missing_woe']

    if rule['kind'] == 'numeric':
        idx = int(np.searchsorted(rule['edges'], float(value), side='right'))
        return rule['woe'][idx]

    return rule['woe'].get(str(value), rule['unseen_woe'])


def transform(X, spec):
    out = {}
    for col, rule in spec.items():
        s = X[col]
        if rule['kind'] == 'numeric':
            v = s.to_numpy(dtype=float)
            idx = np.searchsorted(rule['edges'], v, side='right')
            woe = np.asarray(rule['woe'], dtype=float)[idx]
            out[col] = np.where(pd.isna(v), rule['missing_woe'], woe)
        else:
            mapped = s.astype('object').map(
                lambda v: rule['woe'].get(str(v), rule['unseen_woe'])
                if pd.notna(v) else rule['missing_woe'])
            out[col] = mapped.to_numpy(dtype=float)
    return pd.DataFrame(out, index=X.index)
