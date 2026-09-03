# Credit risk - application scoring

A WOE scorecard on Home Credit Default Risk. The service takes an application
form and returns a probability of default, a decision, and the reasons behind
a decline.

## Task

For a new applicant, knowing only the fields they filled in themselves:

1. estimate the probability of default;
2. approve or decline;
3. on a decline, name the features that caused it.

`EXT_SOURCE_1/2/3` (ready-made external bureau scores) are excluded: the
applicant does not enter them, and an opaque score cannot be presented as a
reason for a decline.

## Data

`application_train.csv`, 307,511 applications, 122 columns, plus three credit
history aggregates from `bureau.csv`. `TARGET = 1` means a late payment beyond
N days on one of the first instalments, 8.07% of the sample.

Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
Not part of the repository - 2.5 GB, seven files above the GitHub 100 MB limit.

```bash
kaggle competitions download -c home-credit-default-risk -p datasets/
unzip datasets/home-credit-default-risk.zip -d datasets/
```

The trained scorecard sits in `artifacts/scorecard.json` (12 KB), so the service
runs without the dataset. The dataset is only needed for `scripts/train.py`.

## Feature selection

Three filters in sequence, 117 -> 14, plus three credit history features.

**1. IV > 0.02.** 49 survive.

```
IV = sum over bins of (share_of_good_i - share_of_bad_i) * WOE_i
```

**2. Product criterion.** 14 survive. Removed:

| block | features | reason |
|---|---|---|
| building characteristics | 27 | external registry data, the applicant does not know it |
| bank internal data | 6 | region rating, dates of document and phone changes |
| derived from the address | 3 | address mismatches, region population |
| duplicate region rating | 1 | `r = 0.95` with the other one |

**3. Coefficient sign check.** On WOE every coefficient must be negative. One
that breaks out signals a duplicate or an inverted direction.

- `FLAG_EMP_PHONE` removed: it matches the employment sentinel in 99.996% of
  rows, and the correlation matrix did not reveal the duplicate.
- `AMT_CREDIT` kept (`r = 0.987` with `AMT_GOODS_PRICE`). Its sign was fixed by
  imposing a monotonic direction, see below.

The final model has no wrong-signed coefficients.

**Returned against its IV:** `AMT_INCOME_TOTAL` (`IV = 0.011`). Income is
mandatory for the regulatory affordability assessment and serves as the
denominator of `DTI`.

### Final 17 features

| feature | IV | score range |
|---|---|---|
| `DAYS_EMPLOYED` | 0.1139 | 19.0 |
| `OCCUPATION_TYPE` | 0.0816 | 20.1 |
| `FIRST_CREDIT_DAYS` | 0.0765 | 26.8 |
| `OVERPAY` | 0.0723 | 20.7 |
| `NAME_INCOME_TYPE` | 0.0556 | 5.8 |
| `TERM_MONTHS` | 0.0405 | 18.0 |
| `CODE_GENDER` | 0.0373 | 24.0 |
| `N_ACTIVE_CREDITS` | 0.0247 | 23.9 |
| `NAME_FAMILY_STATUS` | 0.0231 | 22.4 |
| `CURRENT_DEBT` | 0.0146 | 9.7 |
| `AMT_INCOME_TOTAL` | 0.0113 | 10.0 |
| `DTI` | 0.0068 | 17.0 |
| `FLAG_OWN_CAR` | 0.0067 | 8.2 |
| `AMT_ANNUITY` | 0.0064 | 11.6 |
| `AMT_CREDIT` | 0.0058 | 5.9 |
| `AMT_GOODS_PRICE` | 0.0027 | 1.6 |
| `FLAG_OWN_REALTY` | 0.0003 | 0.1 |

Score range is how far a feature can move the decision at all:
`Factor * |b_i| * (max WOE_i - min WOE_i)`. It does not follow the IV order:
income spans 10.0 points against 19.0 for employment tenure, so a high income
cannot offset a short tenure.

### Credit history

Three aggregates from `bureau.csv`, all reported by the applicant from memory:

| feature | meaning |
|---|---|
| `N_ACTIVE_CREDITS` | how many credits are open right now |
| `CURRENT_DEBT` | total outstanding balance on them |
| `FIRST_CREDIT_DAYS` | when the first credit was taken, days before the application |

Two cases stay apart: no history at all (`NaN`, 7.88% default rate) and history
with no active credits (zero debt, **5.72%**, the safest group).

The block adds 0.055 Gini, and history depth rather than debt size carries it:

| set | ROC-AUC | Gini |
|---|---|---|
| 14 form features | 0.6539 | 0.308 |
| + `CURRENT_DEBT` only | 0.6598 | 0.320 |
| **+ all three** | **0.6820** | **0.364** |

Measured on the actual values from `bureau.csv`. A live applicant understates
the debt and recalls the first credit date only approximately, so in production
the block will give less.

Derived features, computed from fields already filled in:

```
DTI         = AMT_ANNUITY / AMT_INCOME_TOTAL * 100
TERM_MONTHS = AMT_CREDIT / AMT_ANNUITY
OVERPAY     = (AMT_CREDIT - AMT_GOODS_PRICE) / AMT_GOODS_PRICE
```

## Model

Logistic regression on WOE encoding.

```
WOE_i = ln( share_of_good_i / share_of_bad_i )   smoothing (n + 0.5) / (N + k * 0.5)

logit P(default) = b0 + sum of b_i * WOE_i
```

Cut points: 20 quantile pre-bins, then greedy merging of adjacent pairs until
the default rate moves monotonically in the required direction. Bins holding
less than 1% of the sample are merged into a neighbour.

Features with fewer than 50 distinct values are pre-binned on the value grid
rather than on quantiles. Quantiles cannot reach the sparse tail of a skewed
discrete feature: the 95th percentile of the active-credit count is 5 while the
tail runs to 32, so everything above 5 would collapse into a single bin.

Directions are set in `config.MONOTONIC_DIRECTION`:

| features | default rate trend |
|---|---|
| `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `DTI`, `OVERPAY`, `DAYS_EMPLOYED`, `N_ACTIVE_CREDITS`, `CURRENT_DEBT`, `FIRST_CREDIT_DAYS` | ascending |
| `AMT_INCOME_TOTAL`, `TERM_MONTHS` | descending |

In the data larger loans and payments carry a lower default rate: LTI 10+ shows
6.59% against 8.77% for LTI 2-4. Such loans were granted to more thoroughly
vetted clients, so the loan size encodes the depth of screening rather than
affordability.

Monotonicity costs 0.011 AUC against unconstrained quantile binning.

Cut points and WOE are frozen on train and reused on valid. The split is 75/25,
stratified, and happens before the rare-category map is computed.

## Metrics

Evaluated on the holdout, 76,878 applications.

```
ROC-AUC = P(score of a random defaulter > score of a random non-defaulter)
Gini    = 2 * ROC-AUC - 1
PR-AUC  = area under the precision-recall curve; baseline = share of defaults
Brier   = (1/n) * sum of (p_i - y_i)^2
```

| metric | value |
|---|---|
| ROC-AUC | 0.6820 |
| Gini | 0.3641 |
| PR-AUC | 0.1664 (baseline 0.0807) |
| Brier | 0.07142 |
| approval rate | 92.6% |
| default rate among approved | 6.94% (base 8.07%) |

Calibration: ECE 0.0039, the mean predicted PD matches the actual 0.0807. No
class weighting and no resampling: the logistic regression is fitted by maximum
likelihood on the same distribution and calibrates itself.

Isotonic calibration on top is available behind `--calibrate`, with a three-way
split and the calibrator fitted on a separate sample. It does not pay off:

| | without | with calibration |
|---|---|---|
| train | 230,633 | 184,506 |
| ECE | 0.00465 | 0.00419 |
| ROC-AUC | 0.6537 | 0.6523 |
| PR-AUC | 0.1506 | 0.1445 |

Isotonic regression is a step function: it collapses distinct scores into equal
ones and loses ranking resolution. Off by default.

## Approval threshold

Approve while the expectation is positive:

```
(1 - p) * margin > p * LGD

p* = margin / (margin + LGD)
```

The loan amount cancels out: both margin and LGD are proportional to exposure,
so one threshold serves everyone.

```
margin = 0.1367   median positive markup, (AMT_CREDIT - AMT_GOODS_PRICE) / AMT_CREDIT
LGD    = 0.70     assumption, absent from the data
p*     = 0.1634
```

`LGD` is not in the dataset, there are no recovery rates and no collection data.
Sensitivity:

| LGD | p* | approved | default among approved |
|---|---|---|---|
| 0.5 | 0.2148 | 98.7% | 7.82% |
| 0.6 | 0.1856 | 96.9% | 7.55% |
| 0.7 | 0.1634 | 94.3% | 7.30% |
| 0.8 | 0.1460 | 91.1% | 6.99% |

The empirical profit maximum on the holdout sits at 0.150 against 0.163 from the
formula, and the profit curve is flat near the maximum. The gap comes from using
a single median margin while the actual markup varies per application and
correlates with risk.

## Policy

Hard cutoffs applied before the model. The model is trained on granted loans and
does not extrapolate beyond that range: binning clamps extreme values, so a 4M
request against a 25k income lands in the same bins as an ordinary one.

```
income < 20,000              payment >= income
DTI > 50%                    LTI > 10
term outside 8-48 months     more than 10 active credits
```

Cutoffs follow the range of the data. The sample minimum term is 8.0 months and
there are no shorter applications. Above 6 active credits the binning stops
resolving, and only 870 applicants out of 307k hold 10 or more, too few to
estimate a rate for.

The response distinguishes `refused_by: "policy"` from `"scorecard"`.

## Decline reasons

The score decomposes across features:

```
Factor = PDO / ln(2) = 20 / ln 2 = 28.85
Offset = 600 - Factor * ln(50) = 487.1

points_i = Offset/n - Factor * b0/n - Factor * b_i * WOE_i
Score    = sum of points_i = Offset + Factor * ln(odds of being good)
```

600 points corresponds to odds of 50:1, and every +20 points doubles that ratio.

A reason is how many points were lost against the typical applicant:

```
lost_i = points_i(mean WOE over the population) - points_i(actual)
```

The base cancels in the difference. Sorted descending, the top three are the
decline reasons.

`relative_to='max'` measures against the best bin of a feature. 59.5% of
applicants fall into the worst `TERM_MONTHS` bin, so under that convention the
term shows up as a reason for the majority.

## Limitations

- **Selective labels.** The data holds granted loans only. The model is valid on
  a region resembling the approved population.
- **Out-of-time validation is impossible**, the dataset carries no calendar date,
  only relative `DAYS_*`.
- **`OVERPAY` is endogenous.** The markup is set by the bank from its own risk
  assessment. Whether it is known to the applicant before submission needs
  checking.
- **`CODE_GENDER` is prohibited** in credit decisions (EU, ECOA). Switch it off
  with `DROP_PROHIBITED = True` in `src/config.py`, it costs 0.037 IV.
- **Reported income barely predicts default.** Across the bottom 70% of the
  distribution the rate is flat, 8.2-8.9%. It works only in ratios.
- **`FLAG_OWN_REALTY` carries no signal**, IV 0.0003, 8.32% default against 7.96%.

## Running

```bash
pip install -r requirements.txt

python scripts/train.py            # -> artifacts/scorecard.json
uvicorn app.main:app --reload      # -> localhost:8000
```

Training flags: `--binning quantile`, `--calibrate`, `--lgd 0.6`, `--out PATH`.

### API

```
POST /api/score     form -> pd, score, approved, refused_by, violations, reasons
GET  /api/limits    policy limits
GET  /api/health    status, threshold, Gini
```

## Static build

`static/` holds the same scorecard ported to JavaScript: the model is the 12 KB
`scorecard.json`, and scoring is a lookup over cut points plus arithmetic, so it
runs entirely in the browser with no backend.

Verified against the Python model on four cases: probability, score, policy
violations and the reason ordering agree to 1e-9.

```bash
python -m http.server 8010 --directory static
```

Deployed as a Hugging Face static Space; the FastAPI service in `app/` stays the
reference implementation.

## Layout

```
src/
  config.py      features, thresholds, monotonic directions, LGD, score scaling
  data.py        loading, cleaning, derived features, splitting, rare categories
  binning.py     binning and WOE, specification as a plain dict
  scorecard.py   Scorecard: fit, predict_proba, points, reasons, save/load
  policy.py      cutoffs applied before the model
scripts/train.py training and artifact export
app/main.py      FastAPI: form and API
notebooks/       exploration, EDA, the reasoning behind the decisions
artifacts/       scorecard.json - cut points, WOE, coefficients, threshold
static/          browser-only build: the same scorecard ported to JavaScript
Dockerfile       serving image, requirements-serve.txt only
```

`scorecard.json` is 12 KB and holds the model in full. Inference is a lookup
over cut points plus arithmetic.

The notebook and the pipeline differ slightly in their numbers: the notebook
drops the six rows with junk categories while the pipeline maps them to `NaN`.
The user interface is in Russian, the code and documentation are in English.

## License

BSD 2-Clause. See `LICENSE`.
