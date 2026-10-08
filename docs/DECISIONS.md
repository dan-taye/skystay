# Decisions

Format: date · decision · reason · alternative considered.

| Date | Decision | Reason | Alternative |
| :- | :- | :- | :- |
| 2026-10-09 | Run locally on the team's Windows PC (OneDrive folder) instead of a cloud session; git history is pushed to a private GitHub repo once it exists. | The build was started on a Windows PC; there was no repo and no GitHub CLI. | Cloud session or codespace (still works: `skystay all` from a fresh clone). |
| 2026-10-09 | Python venv lives outside OneDrive (`%USERPROFILE%\.venvs\skystay`), not in `.venv/`. | OneDrive would sync ~1 GB / tens of thousands of venv files and can lock files during `pip install`. `.venv/` is still git-ignored and is the documented location for codespaces. | `.venv/` in the project folder. |
| 2026-10-09 | Python 3.13 (not the system 3.14). | Wheels for numba/shap/lightgbm are reliable on 3.13. | 3.14. |
| 2026-10-09 | DATA_MODE=synthetic. | `data/raw/` is empty: no real data was provided. | - |
| 2026-10-09 | ffmpeg comes from the `imageio-ffmpeg` pip package; video metadata is read by parsing `ffmpeg -i` because the bundle has no ffprobe. | ffmpeg is not installed system-wide and installing system packages on a personal PC needs permission. | winget install of ffmpeg. |
| 2026-10-09 | Synthetic scales are round arbitrary numbers, not calibrated to published Abu Dhabi statistics. | Keeps the synthetic data unmistakably synthetic; only the method is being demonstrated. See data/synthetic/SOURCES.md. | Calibrate magnitudes to official DCT / airport statistics. |
| 2026-10-09 | Synthetic window 2019-01..2025-12 (84 months). | Gives three rolling-origin folds (test 2022, 2023, 2024) with >= 36 training months each, plus a 2025 holdout. | 60 months (only 1-2 folds with >= 36 months). |
| 2026-10-09 | Transfer share comes from the passenger-traffic transfer column (transfer / arriving at airport x month). | That is the connection indicator in the data; the schedule has no per-flight connection flag. | Assume a fixed share per market. |
| 2026-10-09 | Load factor is computed at airport x month (finest grain shared by schedule and traffic); LF > 1 flagged, clipped at 1.05. | As specified in the data-pipeline skill. | Route-level LF (traffic is not reported per airline). |
| 2026-10-09 | `skystay all` runs backtest before train. | The production model uses the reconciliation level and model chosen by the back-test. `skystay train` alone reuses the last selection in reports/metrics.json (or defaults to L1 + hybrid if none exists). | Train first with a fixed choice. |
| 2026-10-09 | Robust-z outlier check floors the MAD at 0.05 log units. | Discrete weekly frequencies (e.g. 4/week) give a near-zero MAD and flag ordinary schedule wobble. | Unfloored MAD. |
