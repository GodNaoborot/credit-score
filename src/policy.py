# -*- coding: utf-8 -*-
"""Правила политики - жёсткие отсечения до скоркарты.

Модель обучена на выданных кредитах, то есть на заявках, уже прошедших отбор
банка. Внутри этой популяции она ранжирует корректно, но за её пределами
экстраполировать не может: биннинг клампит крайние значения, и заявка на
4 млн при доходе 25 тысяч попадает в те же корзины, что обычная.

Политика отвечает на вопрос «допустима ли заявка вообще», скоркарта -
на вопрос «насколько рискованна допустимая заявка». Это разные вопросы.
"""
from . import config as cfg

MAX_DTI = 50.0          # платёж к доходу, %
MAX_LTI = 10.0          # сумма кредита к годовому доходу
MIN_INCOME = 20000.0

# Границы срока взяты по диапазону обучающих данных, а не по здравому смыслу:
# минимум в выборке 8.0 месяцев, максимум 45.3, заявок короче 8 месяцев нет
# вообще. За этими границами модель экстраполирует в пустоту, а TERM_MONTHS -
# самый весомый признак скоркарты (размах 22.6 балла).
MIN_TERM_MONTHS = 8.0
MAX_TERM_MONTHS = 48.0


def check(row):
    """Возвращает список нарушений. Пустой список - заявка допустима."""
    income = row.get('AMT_INCOME_TOTAL') or 0
    credit = row.get('AMT_CREDIT') or 0
    annuity = row.get('AMT_ANNUITY') or 0

    violations = []

    for field, text in (('AMT_CREDIT', 'сумма кредита должна быть больше нуля'),
                        ('AMT_ANNUITY', 'ежемесячный платёж должен быть больше нуля'),
                        ('AMT_GOODS_PRICE', 'цена товара должна быть больше нуля')):
        value = row.get(field)
        if value is None or value <= 0:
            violations.append({'rule': f'{field}_POSITIVE', 'text': text})

    if violations:
        return violations

    if income < MIN_INCOME:
        violations.append({
            'rule': 'MIN_INCOME',
            'text': f'доход ниже минимального ({MIN_INCOME:,.0f})',
        })

    if income > 0 and annuity >= income:
        violations.append({
            'rule': 'PAYMENT_OVER_INCOME',
            'text': 'ежемесячный платёж не меньше месячного дохода',
        })

    if income > 0:
        dti = annuity / income * 100
        if dti > MAX_DTI:
            violations.append({
                'rule': 'MAX_DTI',
                'text': f'долговая нагрузка {dti:.0f}% выше предельных {MAX_DTI:.0f}%',
            })

        lti = credit / (income * 12)
        if lti > MAX_LTI:
            violations.append({
                'rule': 'MAX_LTI',
                'text': f'сумма кредита превышает годовой доход в {lti:.1f} раза '
                        f'при пределе {MAX_LTI:.0f}',
            })

    if annuity > 0:
        term = credit / annuity
        if term > MAX_TERM_MONTHS:
            violations.append({
                'rule': 'MAX_TERM',
                'text': f'срок {term:.0f} месяцев выше предельных {MAX_TERM_MONTHS:.0f}',
            })
        elif term < MIN_TERM_MONTHS:
            violations.append({
                'rule': 'MIN_TERM',
                'text': f'срок {term:.0f} месяцев ниже минимальных {MIN_TERM_MONTHS:.0f}',
            })

    return violations


def limits():
    return {'max_dti': MAX_DTI, 'max_lti': MAX_LTI, 'min_income': MIN_INCOME,
            'term_months': [MIN_TERM_MONTHS, MAX_TERM_MONTHS]}
