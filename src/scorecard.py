# -*- coding: utf-8 -*-
"""Scorecard: fitting, score scaling, serialisation.

A fitted scorecard fits entirely into JSON: cut points, WOE, coefficients,
score scaling and the threshold. Inference needs neither sklearn nor pandas,
only arithmetic and a lookup over the cut points.
"""
import json
import math
from datetime import date

import numpy as np

from . import binning
from . import config as cfg


class Scorecard:

    def __init__(self, spec, coef, intercept, features, meta=None, calibrator=None):
        self.spec = spec
        self.coef = dict(coef)
        self.intercept = float(intercept)
        self.features = list(features)
        self.meta = meta or {}
        self.calibrator = calibrator

        self.factor = cfg.PDO / math.log(2)
        self.offset = cfg.BASE_SCORE - self.factor * math.log(cfg.BASE_ODDS)

    # ---------- inference ----------

    def woe_of(self, row):
        return {c: binning.transform_value(self.spec[c], row.get(c)) for c in self.features}

    def log_odds(self, row):
        """Log-odds of default."""
        woe = self.woe_of(row)
        return self.intercept + sum(self.coef[c] * woe[c] for c in self.features)

    def raw_proba(self, row):
        return 1.0 / (1.0 + math.exp(-self.log_odds(row)))

    def predict_proba(self, row):
        """Calibrated probability of default."""
        return self.calibrate(self.raw_proba(row))

    def calibrate(self, p):
        if not self.calibrator:
            return p
        return float(np.interp(p, self.calibrator['x'], self.calibrator['y']))

    def points(self, row):
        """Points per feature. More points means safer. They sum to the total score."""
        woe = self.woe_of(row)
        n = len(self.features)
        base = self.offset / n - self.factor * self.intercept / n
        return {c: base - self.factor * self.coef[c] * woe[c] for c in self.features}

    def score(self, row):
        return sum(self.points(row).values())

    def _all_woe(self, feature):
        rule = self.spec[feature]
        values = list(rule['woe']) if rule['kind'] == 'numeric' else list(rule['woe'].values())
        return values + [rule['missing_woe']]

    def max_points(self, feature):
        """Best attainable points for a feature: base plus its best bin."""
        n = len(self.features)
        base = self.offset / n - self.factor * self.intercept / n
        best_woe = max(self._all_woe(feature)) if self.coef[feature] < 0 \
            else min(self._all_woe(feature))
        return base - self.factor * self.coef[feature] * best_woe

    def reasons(self, row, top=3, relative_to='population'):
        """Decline reasons: how many points were lost.

        relative_to='max'        against the feature's best bin
        relative_to='population' against the typical applicant

        The first convention answers "how far below the ideal are you", the
        second "where are you worse than usual". The second is more useful: 60%
        of applicants fall into the worst TERM_MONTHS bin, so against the ideal
        the term shows up as a reason for almost everyone and explains nothing.

        The base cancels in the difference, so the numbers compare across features.
        """
        pts = self.points(row)
        if relative_to == 'max':
            reference = {c: self.max_points(c) for c in self.features}
        else:
            reference = self.reference_points()

        lost = [{'feature': c,
                 'lost': round(reference[c] - pts[c], 1),
                 'points': round(pts[c], 1)}
                for c in self.features]
        lost.sort(key=lambda d: -d['lost'])
        lost = [d for d in lost if d['lost'] > 0] or lost
        return lost[:top] if top else lost

    def reference_points(self):
        """Points of the typical applicant: mean WOE over the training population."""
        mean_woe = self.meta.get('mean_woe', {})
        n = len(self.features)
        base = self.offset / n - self.factor * self.intercept / n
        return {c: base - self.factor * self.coef[c] * mean_woe.get(c, 0.0)
                for c in self.features}

    def decision(self, row, threshold=None):
        p = self.predict_proba(row)
        thr = self.meta.get('threshold', 0.5) if threshold is None else threshold
        return {'pd': p, 'score': round(self.score(row)), 'approved': bool(p < thr),
                'threshold': thr}

    # ---------- serialisation ----------

    def to_dict(self):
        return {
            'version': 1,
            'created': date.today().isoformat(),
            'features': self.features,
            'spec': self.spec,
            'coef': self.coef,
            'intercept': self.intercept,
            'calibrator': self.calibrator,
            'points': {'pdo': cfg.PDO, 'base_score': cfg.BASE_SCORE,
                       'base_odds': cfg.BASE_ODDS},
            'meta': self.meta,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['spec'], d['coef'], d['intercept'], d['features'],
                   d.get('meta'), d.get('calibrator'))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding='utf-8')
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(path.read_text(encoding='utf-8')))


def approval_threshold(margin, lgd=cfg.LGD):
    """p* = margin / (margin + LGD). The loan amount cancels out."""
    return margin / (margin + lgd)


def estimate_margin(X):
    """Median positive markup as a proxy for gross margin."""
    markup = (X['AMT_CREDIT'] - X['AMT_GOODS_PRICE']) / X['AMT_CREDIT']
    return float(markup[markup > 0].median())


def fit(X_train, y_train, features, categorical, method=None):
    from sklearn.linear_model import LogisticRegression

    spec = binning.fit(X_train, y_train, features, categorical, method)
    W = binning.transform(X_train, spec)[features]

    model = LogisticRegression(max_iter=1000).fit(W, y_train)
    coef = dict(zip(features, model.coef_[0].astype(float)))

    return Scorecard(spec, coef, float(model.intercept_[0]), features)


def raw_proba_batch(card, X):
    W = binning.transform(X, card.spec)[card.features]
    z = card.intercept + W.to_numpy() @ np.array([card.coef[c] for c in card.features])
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba_batch(card, X):
    p = raw_proba_batch(card, X)
    if not card.calibrator:
        return p
    return np.interp(p, card.calibrator['x'], card.calibrator['y'])


def fit_calibrator(card, X_calib, y_calib):
    """Isotonic regression over predictions on data the model has not seen."""
    from sklearn.isotonic import IsotonicRegression

    p = raw_proba_batch(card, X_calib)
    iso = IsotonicRegression(out_of_bounds='clip').fit(p, np.asarray(y_calib))
    card.calibrator = {'x': [float(v) for v in iso.X_thresholds_],
                       'y': [float(v) for v in iso.y_thresholds_]}
    return card
