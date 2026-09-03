# -*- coding: utf-8 -*-
"""Сервис скоринга.

    uvicorn app.main:app --reload

Модель загружается из artifacts/scorecard.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src import config as cfg
from src import policy
from src.scorecard import Scorecard

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))

CARD = Scorecard.load(cfg.ARTIFACT_DIR / 'scorecard.json')

LABELS = {
    'CODE_GENDER': 'Пол',
    'NAME_FAMILY_STATUS': 'Семейное положение',
    'FLAG_OWN_REALTY': 'Есть жильё в собственности',
    'FLAG_OWN_CAR': 'Есть автомобиль',
    'OCCUPATION_TYPE': 'Профессия',
    'NAME_INCOME_TYPE': 'Источник дохода',
    'DAYS_EMPLOYED': 'Стаж на текущем месте',
    'AMT_INCOME_TOTAL': 'Доход',
    'AMT_CREDIT': 'Сумма кредита',
    'AMT_GOODS_PRICE': 'Цена товара',
    'AMT_ANNUITY': 'Ежемесячный платёж',
    'N_ACTIVE_CREDITS': 'Активных кредитов',
    'CURRENT_DEBT': 'Текущий долг',
    'FIRST_CREDIT_DAYS': 'Давность первого кредита',
    'DTI': 'Долговая нагрузка',
    'TERM_MONTHS': 'Срок, месяцев',
    'OVERPAY': 'Наценка',
}

CATEGORICAL = [c for c in cfg.CATEGORICAL if c in CARD.features]

# признак модели -> имя поля формы
FIELD_TO_FORM = {
    'CODE_GENDER': 'gender',
    'NAME_FAMILY_STATUS': 'family_status',
    'FLAG_OWN_REALTY': 'own_realty',
    'FLAG_OWN_CAR': 'own_car',
    'OCCUPATION_TYPE': 'occupation',
    'NAME_INCOME_TYPE': 'income_type',
}


def options(field):
    return sorted(CARD.spec[field]['woe'])


class Application(BaseModel):
    years_employed: float | None = None
    n_active_credits: float = 0
    current_debt: float | None = None
    years_since_first_credit: float | None = None
    income: float
    credit: float
    goods_price: float
    annuity: float
    gender: str = 'F'
    family_status: str = 'Married'
    own_realty: str = 'Y'
    own_car: str = 'N'
    occupation: str = 'Laborers'
    income_type: str = 'Working'


def to_row(a: Application) -> dict:
    """Анкета -> признаки модели. Производные считаются здесь же."""
    row = {
        'CODE_GENDER': a.gender,
        'NAME_FAMILY_STATUS': a.family_status,
        'FLAG_OWN_REALTY': a.own_realty,
        'FLAG_OWN_CAR': a.own_car,
        'OCCUPATION_TYPE': a.occupation,
        'NAME_INCOME_TYPE': a.income_type,
        'DAYS_EMPLOYED': None if a.years_employed is None else -a.years_employed * 365.25,
        'AMT_INCOME_TOTAL': a.income,
        'AMT_CREDIT': a.credit,
        'AMT_GOODS_PRICE': a.goods_price,
        'AMT_ANNUITY': a.annuity,
        'N_ACTIVE_CREDITS': a.n_active_credits,
        'CURRENT_DEBT': a.current_debt,
        'FIRST_CREDIT_DAYS': (None if a.years_since_first_credit is None
                              else -a.years_since_first_credit * 365.25),
    }
    row['DTI'] = a.annuity / a.income * 100 if a.income else None
    row['TERM_MONTHS'] = a.credit / a.annuity if a.annuity else None
    row['OVERPAY'] = (a.credit - a.goods_price) / a.goods_price if a.goods_price else None
    return {c: row.get(c) for c in CARD.features}


app = FastAPI(title='Credit scoring')


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, 'index.html', {
        'labels': LABELS,
        'categorical': CATEGORICAL,
        'options': {c: options(c) for c in CATEGORICAL},
        'field_to_form': FIELD_TO_FORM,
        'meta': CARD.meta,
        'result': None,
    })


@app.post('/', response_class=HTMLResponse)
def score(
    request: Request,
    income: float = Form(...),
    credit: float = Form(...),
    goods_price: float = Form(...),
    annuity: float = Form(...),
    years_employed: str = Form(''),
    n_active_credits: float = Form(0),
    current_debt: str = Form(''),
    years_since_first_credit: str = Form(''),
    gender: str = Form('F'),
    family_status: str = Form('Married'),
    own_realty: str = Form('Y'),
    own_car: str = Form('N'),
    occupation: str = Form('Laborers'),
    income_type: str = Form('Working'),
):
    a = Application(
        years_employed=float(years_employed) if years_employed.strip() else None,
        n_active_credits=n_active_credits,
        current_debt=float(current_debt) if current_debt.strip() else None,
        years_since_first_credit=(float(years_since_first_credit)
                                  if years_since_first_credit.strip() else None),
        income=income, credit=credit, goods_price=goods_price, annuity=annuity,
        gender=gender, family_status=family_status, own_realty=own_realty,
        own_car=own_car, occupation=occupation, income_type=income_type,
    )
    row = to_row(a)
    violations = policy.check(row)
    result = CARD.decision(row)
    result['violations'] = violations
    if violations:
        result['approved'] = False
        result['refused_by'] = 'policy'
    else:
        result['refused_by'] = None if result['approved'] else 'scorecard'
    result['reasons'] = [{'label': LABELS.get(r['feature'], r['feature']), **r}
                         for r in CARD.reasons(row, top=None)]
    result['max_lost'] = max((r['lost'] for r in result['reasons']), default=1) or 1
    result['derived'] = [
        text for text in (
            None if row['DTI'] is None else f"нагрузка {row['DTI']:.1f}%",
            None if row['TERM_MONTHS'] is None else f"срок {row['TERM_MONTHS']:.0f} мес",
            None if row['OVERPAY'] is None else f"наценка {row['OVERPAY'] * 100:.1f}%",
        ) if text
    ]

    return templates.TemplateResponse(request, 'index.html', {
        'labels': LABELS,
        'categorical': CATEGORICAL,
        'options': {c: options(c) for c in CATEGORICAL},
        'field_to_form': FIELD_TO_FORM,
        'meta': CARD.meta,
        'result': result,
        'form': a.model_dump(),
    })


@app.post('/api/score')
def api_score(a: Application):
    row = to_row(a)
    violations = policy.check(row)
    out = CARD.decision(row)
    out['violations'] = violations
    if violations:
        out['approved'] = False
        out['refused_by'] = 'policy'
    else:
        out['refused_by'] = None if out['approved'] else 'scorecard'
    out['reasons'] = CARD.reasons(row, top=3)
    return out


@app.get('/api/limits')
def limits():
    return policy.limits()


@app.get('/api/health')
def health():
    return {'status': 'ok', 'features': len(CARD.features),
            'threshold': CARD.meta.get('threshold'),
            'gini': CARD.meta.get('metrics', {}).get('gini')}
