# SkyStay: flight-to-hotel demand scenario simulator

A personal project by Daniel: a flight-to-hotel demand simulator for Abu Dhabi tourism planning.

SkyStay links Abu Dhabi air capacity to hotel demand:

**seats → passengers (load factor) → visitors (minus transfer/transit) → hotel guests by nationality (learned origin-to-nationality bridge × capture rate + non-air demand) → guest nights (length of stay)**

A planner changes a lever (a new route, more or fewer weekly flights, bigger aircraft, load factor, transfer share, capture rate, length of stay, market mix, seasonal shifts, events). Hotel guests and nights then update by source market and month, with 80% and 95% ranges. The tool also shows which levers matter most and produces a plain-English Planner Brief.

> **Data mode.** If `data/raw/` contains real data files, `DATA_MODE=real`; otherwise the pipeline runs on a seeded **synthetic** stand-in with known ground truth, and every page, chart, slide and video frame says **SYNTHETIC DEMO DATA**. Synthetic results demonstrate the method only; they are not claims about real Abu Dhabi demand.

## Quickstart

```bash
python -m venv .venv
```

Activate the venv (Windows: `.venv\Scripts\activate`; macOS/Linux/codespace: `source .venv/bin/activate`). Then:

```bash
pip install -e .
```

```bash
python -m playwright install chromium
```

```bash
skystay all
```

```bash
skystay app
```

Open http://localhost:8501. In a GitHub codespace, open the forwarded port from the **Ports** tab. Video and deck need Chromium and ffmpeg: ffmpeg comes with the `imageio-ffmpeg` package. Use `skystay all --skip-media` to skip them.

With Docker, which builds the data, model and reports inside the image and serves the app on localhost:8501:

```bash
docker compose up --build
```

### Commands

| Command | Does |
| :- | :- |
| `skystay setup` | environment checks; detects the data mode and writes `config/settings.yaml` |
| `skystay data` | ingest `data/raw` (or generate synthetic data), quality checks, DuckDB store |
| `skystay backtest` | rolling-origin back-test of every model at every reconciliation level; selects the winner |
| `skystay train` | fit the selected model on all data; writes the assumption ledger and model bundle |
| `skystay sensitivity` | tornado, Sobol, guests per 1,000 seats, seasonal vulnerability, Planner Brief, docs |
| `skystay app` | launch the simulator |
| `skystay test` | run the test suite |
| `skystay video` | record and assemble the demo video |
| `skystay deck` | build the 10-page PDF deck |
| `skystay all` | everything end to end |
| `skystay clean` | remove generated outputs |

The Makefile is a thin wrapper around these commands.

## Architecture

```
data/raw (real data)  or  synthetic generator (seed 42)
        │  adapters (config/schema_map.yaml), data-quality checks
        ▼
DuckDB canonical store ── flights_month · pax_month · hotel_month · events · holidays · seasonality · market_map
        ▼
Layer 1  parametric chain (EB-shrunk LF, transfer, capture, ALOS, non-air baseline)
         + country reconciliation: L0 identity · L1 regional · L2 learned bridge
Layer 2  econometric benchmark (OLS, HC3)        Layer 3  LightGBM correction (monotone)
        ▼                                                ▼
        └──────── hybrid = chain × exp(ML correction) ───┘   chosen by back-test (WMAPE)
        ▼
Scenario engine (11 levers, guardrails, Monte Carlo + conformal intervals)
        ▼
Streamlit app (8 pages) · sensitivity · Planner Brief · video · deck
```

Code layout: `src/skystay/{io, reconcile, chain, ml, backtest, scenario, sensitivity, app, video, deck}`, `config/`, `data/`, `reports/`, `docs/`, `tests/`.

## The country catch

Flight data records the **departure country**. Hotel data records the guest's **nationality**. Hubs carry many nationalities, expatriates fly from where they live, residents return home, and some guests arrive by land or via other UAE airports. We compare three ways to reconcile the two in the back-test: identity, regional aggregation, and a learned non-negative origin-to-nationality bridge with bootstrap uncertainty. The bridge heatmap is on the app's *Data & assumptions* page.

## Results

<!-- RESULTS:START -->
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
<!-- RESULTS:END -->

## Assumptions and limitations

- Every assumption, with value, unit, 80% interval, n_obs, method, source and adjustable flag: `config/assumptions.yaml` (generated), readable as `docs/ASSUMPTIONS.md`.
- Limitations: `docs/LIMITATIONS.md`. Model card: `docs/MODEL_CARD.md`. Decisions log: `docs/DECISIONS.md`. Data notes: `docs/DATA_NOTES.md`.
- No leakage: every back-test fold is estimated on months before its cutoff (`tests/test_leakage.py`).

## Using real data

1. Put the raw files (flight schedule, passenger traffic, hotel performance, calendars and their data dictionary) in `data/raw/`.
2. Edit `config/schema_map.yaml` so file patterns and column names match the dictionary. Adapters fail loudly and name any unmapped column or unknown country label; add country aliases to `config/regions.yaml`.
3. Run `skystay all`. The data mode switches to `real` automatically and the synthetic badge disappears.

## Dependency licences

| Package | Licence |
| :- | :- |
| pandas, numpy, scipy, statsmodels, Jinja2 | BSD-3-Clause |
| duckdb, lightgbm, shap, SALib, pydantic, PyYAML, plotly, openpyxl, pytest, ruff | MIT |
| pyarrow, streamlit, playwright | Apache-2.0 |
| pillow | MIT-CMU (HPND) |
| imageio-ffmpeg | BSD-2-Clause (the bundled ffmpeg binary is LGPL/GPL) |

## Notes

- Everything here runs on the synthetic stand-in data.
- The demo video and the PDF deck are in `demo/` (the video has a placeholder synthetic voiceover).
