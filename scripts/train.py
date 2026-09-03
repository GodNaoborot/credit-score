# -*- coding: utf-8 -*-
"""Scorecard training.

    python scripts/train.py
    python scripts/train.py --binning quantile --lgd 0.6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from src import config as cfg
from src import data, scorecard


def evaluate(y, proba, threshold):
    y = np.asarray(y)
    approved = proba < threshold
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy='quantile')
    return {
        'ece': round(float(np.abs(frac_pos - mean_pred).mean()), 5),
        'roc_auc': round(float(roc_auc_score(y, proba)), 4),
        'gini': round(float(2 * roc_auc_score(y, proba) - 1), 4),
        'pr_auc': round(float(average_precision_score(y, proba)), 4),
        'pr_auc_base': round(float(y.mean()), 4),
        'brier': round(float(brier_score_loss(y, proba)), 5),
        'approval_rate': round(float(approved.mean()), 4),
        'bad_rate_approved': round(float(y[approved].mean()), 4),
        'bad_rate_all': round(float(y.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binning', choices=('monotonic', 'quantile'), default=cfg.BINNING)
    ap.add_argument('--lgd', type=float, default=cfg.LGD)
    ap.add_argument('--out', type=Path, default=cfg.ARTIFACT_DIR / 'scorecard.json')
    ap.add_argument('--calibrate', action='store_true',
                    help='isotonic calibration on a held-out sample')
    args = ap.parse_args()

    print('preparing data...')
    X_train, X_calib, X_valid, y_train, y_calib, y_valid, keep = data.prepare(
        calib=args.calibrate)
    features = data.feature_list()
    categorical = [c for c in cfg.CATEGORICAL if c in features]
    calib_shape = X_calib.shape if args.calibrate else '-'
    print(f'  train {X_train.shape}  calib {calib_shape}  valid {X_valid.shape}'
          f'  features {len(features)}')

    print(f'fitting, binning {args.binning}...')
    card = scorecard.fit(X_train, y_train, features, categorical, args.binning)

    raw = scorecard.raw_proba_batch(card, X_valid)
    if args.calibrate:
        scorecard.fit_calibrator(card, X_calib, y_calib)

    margin = scorecard.estimate_margin(X_train)
    threshold = scorecard.approval_threshold(margin, args.lgd)

    proba = scorecard.predict_proba_batch(card, X_valid)
    metrics = evaluate(y_valid, proba, threshold)
    metrics_raw = evaluate(y_valid, raw, threshold)

    odd = [c for c, v in card.coef.items() if v > 0]
    iv = {c: round(card.spec[c]['iv'], 4) for c in features}

    from src import binning as _binning
    mean_woe = _binning.transform(X_train, card.spec)[features].mean().round(6).to_dict()

    card.meta = {
        'binning': args.binning,
        'mean_woe': {k: float(v) for k, v in mean_woe.items()},
        'margin': round(margin, 4),
        'lgd': args.lgd,
        'threshold': round(threshold, 4),
        'rare_categories_kept': keep,
        'calibrated': args.calibrate,
        'metrics': metrics,
        'metrics_uncalibrated': metrics_raw if args.calibrate else None,
        'iv': iv,
        'coef_wrong_sign': odd,
    }

    path = card.save(args.out)

    print('\nmetrics on valid:')
    for k, v in metrics.items():
        suffix = f'   uncalibrated {metrics_raw[k]}' if args.calibrate else ''
        print(f'  {k:20s} {v}{suffix}')
    print(f'\nmargin {margin:.4f}  LGD {args.lgd}  threshold {threshold:.4f}')
    if odd:
        print(f'wrong sign: {odd}')
    print(f'\nsaved: {path}  ({path.stat().st_size / 1024:.0f} KB)')

    # check: row-wise inference must match the batch path
    row = X_valid.iloc[0].to_dict()
    assert abs(card.predict_proba(row) - proba[0]) < 1e-9, 'row-wise inference diverged'
    print('row-wise inference matches the batch path')


if __name__ == '__main__':
    main()
