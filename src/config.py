# -*- coding: utf-8 -*-
"""Константы пайплайна."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'datasets'
ARTIFACT_DIR = ROOT / 'artifacts'

RANDOM_STATE = 42
VALID_SIZE = 0.25      # отложенная выборка, трогается один раз
CALIB_SIZE = 0.20      # доля остатка под калибратор

# ----- чистка -----
SENTINEL_DAYS_EMPLOYED = 365243
RARE_MIN_FREQ = 0.01
JUNK_VALUES = {'CODE_GENDER': ['XNA'], 'NAME_FAMILY_STATUS': ['Unknown']}

# ----- признаки -----
BASE_FEATURES = [
    'CODE_GENDER',
    'NAME_FAMILY_STATUS',
    'FLAG_OWN_REALTY',
    'FLAG_OWN_CAR',
    'OCCUPATION_TYPE',
    'NAME_INCOME_TYPE',
    'DAYS_EMPLOYED',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'AMT_ANNUITY',
    # кредитная история: агрегаты bureau.csv, клиент отвечает по памяти
    'N_ACTIVE_CREDITS',
    'CURRENT_DEBT',
    'FIRST_CREDIT_DAYS',
]
DERIVED_FEATURES = ['DTI', 'TERM_MONTHS', 'OVERPAY']
FEATURES = BASE_FEATURES + DERIVED_FEATURES

CATEGORICAL = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'FLAG_OWN_REALTY',
               'FLAG_OWN_CAR', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE']

# Пол запрещён в кредитных решениях (ЕС, ECOA в США). В ноутбуке оставлен ради
# сопоставимости с прежними замерами; для сервиса переключить на True.
DROP_PROHIBITED = False
PROHIBITED = ['CODE_GENDER']

# ----- биннинг -----
BINNING = 'monotonic'      # 'monotonic' либо 'quantile'
N_BINS = 10                # для 'quantile'
N_PREBINS = 20             # стартовых корзин для 'monotonic'
MIN_BIN_SIZE = 0.03        # минимальная доля выборки в корзине
SMOOTH = 0.5

# Тренд ДОЛИ ДЕФОЛТОВ вдоль признака:
#   'ascending'  больше значение -> выше риск
#   'descending' больше значение -> ниже риск
#
# Направление задано по смыслу. В данных крупные кредиты и платежи имеют более
# низкий дефолт, потому что их выдавали более проверенным клиентам; без
# ограничения модель выучивает обратную зависимость.
MONOTONIC_DIRECTION = {
    'AMT_CREDIT': 'ascending',
    'AMT_ANNUITY': 'ascending',
    'AMT_GOODS_PRICE': 'ascending',
    'DTI': 'ascending',
    'OVERPAY': 'ascending',
    'DAYS_EMPLOYED': 'ascending',      # хранится отрицательным: ближе к нулю = меньше стаж
    'AMT_INCOME_TOTAL': 'descending',
    'TERM_MONTHS': 'descending',       # длиннее срок -> меньше платёж
}
DEFAULT_TREND = 'auto_asc_desc'

# ----- экономика -----
LGD = 0.70                 # допущение: необеспеченный потребкредит, в данных нет

# ----- перевод в баллы -----
PDO = 20                   # на сколько баллов удваиваются шансы
BASE_SCORE = 600
BASE_ODDS = 50
