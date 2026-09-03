# -*- coding: utf-8 -*-
"""Загрузка, чистка, производные признаки, сплит."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg


def load_raw(path=None):
    return pd.read_csv(path or cfg.DATA_DIR / 'application_train.csv')


def clean(df):
    """Правки, не требующие обучения на данных."""
    df = df.copy()

    for col, values in cfg.JUNK_VALUES.items():
        df.loc[df[col].isin(values), col] = np.nan

    df.loc[df['DAYS_EMPLOYED'] == cfg.SENTINEL_DAYS_EMPLOYED, 'DAYS_EMPLOYED'] = np.nan

    return df


def add_derived(X):
    """Производные признаки. Считаются построчно, обучения не требуют."""
    X = X.copy()
    X['DTI'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL'] * 100
    X['TERM_MONTHS'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    X['OVERPAY'] = (X['AMT_CREDIT'] - X['AMT_GOODS_PRICE']) / X['AMT_GOODS_PRICE']
    return X


def feature_list():
    if cfg.DROP_PROHIBITED:
        return [c for c in cfg.FEATURES if c not in cfg.PROHIBITED]
    return list(cfg.FEATURES)


def fit_rare_map(X, columns):
    """Категории, которые остаются как есть. Считается только по train."""
    keep = {}
    for col in columns:
        freq = X[col].value_counts(normalize=True)
        keep[col] = sorted(freq.index[freq >= cfg.RARE_MIN_FREQ].astype(str))
    return keep


def apply_rare_map(X, keep):
    X = X.copy()
    for col, allowed in keep.items():
        s = X[col].astype('object')
        X[col] = s.where(s.isin(allowed) | s.isna(), 'OTHER')
    return X


def prepare(df=None):
    """Полный путь от сырого файла до train/valid."""
    df = clean(load_raw() if df is None else df)

    features = feature_list()
    X = add_derived(df[cfg.BASE_FEATURES])[features]
    y = df['TARGET']

    ok = y.notna()
    X, y = X[ok], y[ok]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=cfg.VALID_SIZE, random_state=cfg.RANDOM_STATE,
        stratify=y, shuffle=True,
    )

    cat = [c for c in cfg.CATEGORICAL if c in features]
    keep = fit_rare_map(X_train, cat)

    return (apply_rare_map(X_train, keep), apply_rare_map(X_valid, keep),
            y_train, y_valid, keep)
