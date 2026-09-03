# -*- coding: utf-8 -*-
"""Loading, cleaning, derived features, splitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg


def load_raw(path=None):
    return pd.read_csv(path or cfg.DATA_DIR / 'application_train.csv')


def load_bureau(path=None):
    """Credit history aggregates per applicant.

    Two cases stay apart: no history at all (NaN) and history with no active
    credits (zero debt). Their default rates differ, 7.83% against 5.78%.
    """
    b = pd.read_csv(path or cfg.DATA_DIR / 'bureau.csv',
                    usecols=['SK_ID_CURR', 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_DEBT',
                             'DAYS_CREDIT'])
    active = b[b['CREDIT_ACTIVE'] == 'Active']

    return pd.DataFrame({
        'N_ACTIVE_CREDITS': active.groupby('SK_ID_CURR').size(),
        'CURRENT_DEBT': active.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum().clip(lower=0),
        'FIRST_CREDIT_DAYS': b.groupby('SK_ID_CURR')['DAYS_CREDIT'].min(),
    }).reindex(b['SK_ID_CURR'].unique())


def attach_bureau(df, bureau=None):
    bureau = load_bureau() if bureau is None else bureau
    joined = df[['SK_ID_CURR']].join(bureau, on='SK_ID_CURR')
    out = df.copy()
    for col in bureau.columns:
        out[col] = joined[col].to_numpy()
    # no bureau record -> zero active credits, debt and history depth unknown
    out['N_ACTIVE_CREDITS'] = out['N_ACTIVE_CREDITS'].fillna(0)
    return out


def clean(df):
    """Fixes that need no fitting on the data."""
    df = df.copy()

    for col, values in cfg.JUNK_VALUES.items():
        df.loc[df[col].isin(values), col] = np.nan

    df.loc[df['DAYS_EMPLOYED'] == cfg.SENTINEL_DAYS_EMPLOYED, 'DAYS_EMPLOYED'] = np.nan

    return df


def add_derived(X):
    """Derived features. Row-wise arithmetic, no fitting involved."""
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
    """Categories that survive as themselves. Computed on train only."""
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


def prepare(df=None, calib=True):
    """Full path from the raw file to train / calib / valid.

    train  binning and coefficients
    calib  isotonic calibration, rows the model has not seen
    valid  evaluation, touched once

    The split comes before the rare-category map, otherwise the held-out parts
    take part in fitting that mapping.
    """
    df = attach_bureau(clean(load_raw() if df is None else df))

    features = feature_list()
    X = add_derived(df[cfg.BASE_FEATURES])[features]
    y = df['TARGET']

    ok = y.notna()
    X, y = X[ok], y[ok]

    X_rest, X_valid, y_rest, y_valid = train_test_split(
        X, y, test_size=cfg.VALID_SIZE, random_state=cfg.RANDOM_STATE,
        stratify=y, shuffle=True,
    )
    if calib:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_rest, y_rest, test_size=cfg.CALIB_SIZE, random_state=cfg.RANDOM_STATE,
            stratify=y_rest, shuffle=True,
        )
    else:
        X_train, y_train = X_rest, y_rest
        X_calib, y_calib = X_rest.iloc[:0], y_rest.iloc[:0]

    cat = [c for c in cfg.CATEGORICAL if c in features]
    keep = fit_rare_map(X_train, cat)

    return (apply_rare_map(X_train, keep), apply_rare_map(X_calib, keep),
            apply_rare_map(X_valid, keep), y_train, y_calib, y_valid, keep)
