# -*- coding: utf-8 -*-
"""Скоркарта: обучение, перевод в баллы, сериализация.

Обученная скоркарта целиком помещается в JSON - границы корзин, WOE,
коэффициенты, калибровка баллов и порог. Инференсу не нужны ни sklearn,
ни pandas: только арифметика и поиск по границам.
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

    # ---------- инференс ----------

    def woe_of(self, row):
        return {c: binning.transform_value(self.spec[c], row.get(c)) for c in self.features}

    def log_odds(self, row):
        """Лог-шансы дефолта."""
        woe = self.woe_of(row)
        return self.intercept + sum(self.coef[c] * woe[c] for c in self.features)

    def raw_proba(self, row):
        return 1.0 / (1.0 + math.exp(-self.log_odds(row)))

    def predict_proba(self, row):
        """Калиброванная вероятность дефолта."""
        return self.calibrate(self.raw_proba(row))

    def calibrate(self, p):
        if not self.calibrator:
            return p
        return float(np.interp(p, self.calibrator['x'], self.calibrator['y']))

    def points(self, row):
        """Баллы по признакам. Больше баллов - надёжнее. Сумма даёт общий балл."""
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
        """Лучшие достижимые баллы по признаку: база плюс лучшая корзина."""
        n = len(self.features)
        base = self.offset / n - self.factor * self.intercept / n
        best_woe = max(self._all_woe(feature)) if self.coef[feature] < 0 \
            else min(self._all_woe(feature))
        return base - self.factor * self.coef[feature] * best_woe

    def reasons(self, row, top=3, relative_to='population'):
        """Причины отказа: сколько баллов потеряно.

        relative_to='max'        - относительно лучшей корзины признака
        relative_to='population' - относительно среднего заявителя

        Первая конвенция отвечает на «насколько вы хуже идеала», вторая - на
        «где вы хуже обычного». Вторая практичнее: по сроку 60% заявителей
        сидят в худшей корзине, и относительно идеала срок попадает в причины
        почти всем, ничего не объясняя.

        База в разности сокращается, поэтому числа сравнимы между признаками.
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
        """Баллы типичного заявителя: средний WOE по обучающей популяции."""
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

    # ---------- сериализация ----------

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
    """p* = маржа / (маржа + LGD). Сумма кредита сокращается."""
    return margin / (margin + lgd)


def estimate_margin(X):
    """Медианная положительная наценка как прокси валовой маржи."""
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
    """Изотоническая регрессия на предсказаниях по данным, которых модель не видела."""
    from sklearn.isotonic import IsotonicRegression

    p = raw_proba_batch(card, X_calib)
    iso = IsotonicRegression(out_of_bounds='clip').fit(p, np.asarray(y_calib))
    card.calibrator = {'x': [float(v) for v in iso.X_thresholds_],
                       'y': [float(v) for v in iso.y_thresholds_]}
    return card
