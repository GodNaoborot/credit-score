---
title: Credit Risk Scoring
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
license: bsd-2-clause
---

# Credit risk scoring

A WOE scorecard on Home Credit Default Risk. The whole model is the 12 KB
`scorecard.json` next to this page: cut points, WOE, coefficients, score scaling
and the approval threshold. Scoring runs in the browser, nothing is sent
anywhere.

ROC-AUC 0.6820, Gini 0.3641 on a 76,878-application holdout.

Source and the training pipeline: https://github.com/GodNaoborot/Credit_risk
