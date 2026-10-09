# Model card: SkyStay flight-to-hotel demand model

> **SYNTHETIC DEMO DATA.** Trained and evaluated on a seeded synthetic stand-in for real tourism data.

## Intended use
Scenario planning for Abu Dhabi tourism: estimate how changes in air capacity and conversion assumptions move hotel guests and guest nights by source market and month, with uncertainty. For planners at DCT and partners. **Not** for: forecasting beyond the reference year, individual-level decisions, or any use of passenger-level data (none is used).

## Model
- Selected: **L2 learned bridge + Hybrid (chain x ML)** (lowest mean cross-validated WMAPE).
- Layer 1 (transparent chain): seats x load factor x (1 - transfer share) = terminating passengers; a learned origin-to-nationality bridge K (non-negative, ridge toward a regional prior) converts them to air-attributable hotel guests by nationality, with a seasonal capture shape and pooled holiday/event uplifts; a non-air baseline (DCT seasonality profile, ridge-penalised trend) is added; guest nights = guests x length of stay. Empirical-Bayes shrinkage by month of year; cold-start from region averages.
- Layer 2 (benchmark): OLS of log guests on log air visitors with month and market fixed effects, holiday and event terms, HC3 errors. Elasticity 0.49 (95% CI 0.47 to 0.51).
- Layer 3: LightGBM on the log residual of Layer 1 (hybrid) with a monotone-increasing constraint on air visitors; features: market, month, holiday days, event days, trend, 12-month lag.
- Intervals: split-conformal on out-of-fold log residuals (per market, region-pooled when thin), combined in scenarios with Monte Carlo over parameter uncertainty.

## Training data
- Window 2019-01 to 2025-12 (84 months); 29 guest nationalities; aggregated monthly data only (see `docs/DATA_NOTES.md`).

## Evaluation (on synthetic data with known ground truth)

> **SYNTHETIC DEMO DATA.** These results are on synthetic data with known ground truth. They show the method works; they are not real-world accuracy.

Selected by rolling-origin back-test: **L2 learned bridge + Hybrid (chain x ML)**.

| Model | CV WMAPE (guests) | Holdout WMAPE (guests) | Holdout WMAPE (nights) |
| :- | -: | -: | -: |
| Seasonal naive | 8.3% | 8.9% | 9.6% |
| Parametric chain | 6.2% | 6.7% | 7.0% |
| Econometric | 7.8% | 8.6% | 8.8% |
| ML only | 7.1% | 6.9% | 7.3% |
| Hybrid (chain x ML) | 5.7% | 6.5% | 6.9% |

- Holdout (2025-01 to 2025-12): WMAPE 6.5% on monthly hotel guests by market, bias -1.1%.
- Conformal intervals on the holdout: 80% covered 79%, 95% covered 91%.
- Non-air share of hotel guests (latest year): 45%.
- Parameter recovery on synthetic ground truth: 7/7 checks pass.
- Biggest drivers of annual guest nights (Sobol total-order): Length of stay (0.37), Seat capacity (frequency x aircraft size) (0.31), Hotel capture rate (0.14).
- Highest hotel guests per 1,000 added seats: United States (558), China (463), Italy (462).

Full details: `reports/backtest.md`, `reports/metrics.json`, `reports/sensitivity.json`, `reports/planner_brief.md`.

## Parameter recovery (synthetic only)

| Check | Statistic | Value | Tolerance | Pass |
| :- | :- | -: | -: | :- |
| load_factor | max_abs | 0.006 | 0.02 | yes |
| transfer_share | max_abs | 0.005 | 0.02 | yes |
| alos | max_abs | 0.010 | 0.05 | yes |
| non_air_share | median_abs | 0.102 | 0.15 | yes |
| bridge_diagonal | median_abs | 0.092 | 0.15 | yes |
| capture_rate | median_abs | 0.088 | 0.25 | yes |
| domestic_is_non_air | value | 1.000 | 0.999 | yes |

## Limitations and ethical considerations
See `docs/LIMITATIONS.md`. Key points: synthetic data until real data is available; air vs non-air demand only weakly identified; returning residents folded into capture; new routes are cold-start estimates without diversion; intervals calibrated on three folds. Aggregated data only; no personal data.

## Reproducibility
`skystay all` regenerates every number from raw files (seed 42). Tests: `skystay test`.
