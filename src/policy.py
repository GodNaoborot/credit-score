# -*- coding: utf-8 -*-
"""Policy rules: hard cutoffs applied before the scorecard.

The model is trained on granted loans, that is on applications that already
passed the bank's own screening. It ranks correctly inside that population but
cannot extrapolate outside it: binning clamps extreme values, so a 4M request
against a 25k income lands in the same bins as an ordinary one.

Policy answers whether an application is admissible at all, the scorecard
answers how risky an admissible application is. Different questions.
"""
from . import config as cfg

MAX_DTI = 50.0          # payment to income, %
MAX_LTI = 10.0          # loan amount to annual income
MIN_INCOME = 20000.0

# Term limits follow the range of the training data rather than common sense:
# the sample minimum is 8.0 months, the maximum 45.3, and there are no
# applications shorter than 8 months at all. Beyond these bounds the model
# extrapolates into emptiness, and TERM_MONTHS carries the widest score range.
MIN_TERM_MONTHS = 8.0
MAX_TERM_MONTHS = 48.0

# Above 6 active credits the binning stops resolving: 870 applicants out of
# 307k hold 10 or more, too few to estimate a rate for. Their observed default
# rate runs 20% to 42%, so the cutoff is both out of range and plainly bad risk.
MAX_ACTIVE_CREDITS = 10


def check(row):
    """Returns the list of violations. An empty list means the application is admissible."""
    income = row.get('AMT_INCOME_TOTAL') or 0
    credit = row.get('AMT_CREDIT') or 0
    annuity = row.get('AMT_ANNUITY') or 0

    violations = []

    for field, text in (('AMT_CREDIT', 'loan amount must be greater than zero'),
                        ('AMT_ANNUITY', 'monthly payment must be greater than zero'),
                        ('AMT_GOODS_PRICE', 'goods price must be greater than zero')):
        value = row.get(field)
        if value is None or value <= 0:
            violations.append({'rule': f'{field}_POSITIVE', 'text': text})

    if violations:
        return violations

    if income < MIN_INCOME:
        violations.append({
            'rule': 'MIN_INCOME',
            'text': f'income below the minimum of {MIN_INCOME:,.0f}',
        })

    if income > 0 and annuity >= income:
        violations.append({
            'rule': 'PAYMENT_OVER_INCOME',
            'text': 'monthly payment is not below monthly income',
        })

    if income > 0:
        dti = annuity / income * 100
        if dti > MAX_DTI:
            violations.append({
                'rule': 'MAX_DTI',
                'text': f'debt burden {dti:.0f}% exceeds the {MAX_DTI:.0f}% limit',
            })

        lti = credit / (income * 12)
        if lti > MAX_LTI:
            violations.append({
                'rule': 'MAX_LTI',
                'text': f'loan amount exceeds annual income {lti:.1f} times, '
                        f'limit is {MAX_LTI:.0f}',
            })

    n_credits = row.get('N_ACTIVE_CREDITS')
    if n_credits is not None and n_credits > MAX_ACTIVE_CREDITS:
        violations.append({
            'rule': 'MAX_ACTIVE_CREDITS',
            'text': f'{int(n_credits)} active credits, the limit is {MAX_ACTIVE_CREDITS}',
        })

    if annuity > 0:
        term = credit / annuity
        if term > MAX_TERM_MONTHS:
            violations.append({
                'rule': 'MAX_TERM',
                'text': f'term of {term:.0f} months exceeds the {MAX_TERM_MONTHS:.0f} limit',
            })
        elif term < MIN_TERM_MONTHS:
            violations.append({
                'rule': 'MIN_TERM',
                'text': f'term of {term:.0f} months is below the minimum of {MIN_TERM_MONTHS:.0f}',
            })

    return violations


def limits():
    return {'max_dti': MAX_DTI, 'max_lti': MAX_LTI, 'min_income': MIN_INCOME,
            'max_active_credits': MAX_ACTIVE_CREDITS,
            'term_months': [MIN_TERM_MONTHS, MAX_TERM_MONTHS]}
