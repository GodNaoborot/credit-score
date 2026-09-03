# -*- coding: utf-8 -*-
"""Pipeline constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'datasets'
ARTIFACT_DIR = ROOT / 'artifacts'

RANDOM_STATE = 42
VALID_SIZE = 0.25      # holdout, touched once
CALIB_SIZE = 0.20      # share of the remainder used for the calibrator

# ----- cleaning -----
SENTINEL_DAYS_EMPLOYED = 365243
RARE_MIN_FREQ = 0.01
JUNK_VALUES = {'CODE_GENDER': ['XNA'], 'NAME_FAMILY_STATUS': ['Unknown']}

# ----- features -----
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
    # credit history: bureau.csv aggregates, the applicant reports them from memory
    'N_ACTIVE_CREDITS',
    'CURRENT_DEBT',
    'FIRST_CREDIT_DAYS',
]
DERIVED_FEATURES = ['DTI', 'TERM_MONTHS', 'OVERPAY']
FEATURES = BASE_FEATURES + DERIVED_FEATURES

CATEGORICAL = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'FLAG_OWN_REALTY',
               'FLAG_OWN_CAR', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE']

# Gender is prohibited in credit decisions (EU, ECOA in the US). Kept in the
# notebook so the measurements stay comparable; switch to True for the service.
DROP_PROHIBITED = False
PROHIBITED = ['CODE_GENDER']

# ----- binning -----
BINNING = 'monotonic'      # 'monotonic' or 'quantile'
N_BINS = 10                # for 'quantile'
N_PREBINS = 20             # starting bins for 'monotonic'
MIN_BIN_SIZE = 0.01        # smallest share of the sample a bin may hold
DISCRETE_MAX_LEVELS = 50   # below this many distinct values, pre-bin on the value grid
SMOOTH = 0.5

# Trend of the DEFAULT RATE along the feature:
#   'ascending'  higher value -> higher risk
#   'descending' higher value -> lower risk
#
# Directions are set by hand. In the data larger loans and payments show a lower
# default rate because they were granted to more thoroughly vetted clients;
# without the constraint the model learns the inverse relationship.
MONOTONIC_DIRECTION = {
    'AMT_CREDIT': 'ascending',
    'AMT_ANNUITY': 'ascending',
    'AMT_GOODS_PRICE': 'ascending',
    'DTI': 'ascending',
    'OVERPAY': 'ascending',
    'DAYS_EMPLOYED': 'ascending',      # stored negative: closer to zero = shorter tenure
    'AMT_INCOME_TOTAL': 'descending',
    'TERM_MONTHS': 'descending',       # longer term -> smaller payment
    'N_ACTIVE_CREDITS': 'ascending',
    'CURRENT_DEBT': 'ascending',
    'FIRST_CREDIT_DAYS': 'ascending',  # negative days: closer to zero = shorter history
}
DEFAULT_TREND = 'auto_asc_desc'

# ----- economics -----
LGD = 0.70                 # assumption: unsecured consumer lending, absent from the data

# ----- score scaling -----
PDO = 20                   # points to double the odds
BASE_SCORE = 600
BASE_ODDS = 50
