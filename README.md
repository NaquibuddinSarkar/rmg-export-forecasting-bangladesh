# RMG Export Forecasting — Bangladesh

A monthly time-series analysis and forecasting framework for Bangladesh ready-made garment (RMG) exports.

## Project contents

- `src/rmg_export_forecasting/models.py`: seasonal-naive, Holt-Winters and SARIMAX forecasts, common-horizon scoring, expanding-window backtesting and deterministic scenario shocks.
- `scripts/run_analysis.py`: compares candidate models on a chronological 12-month holdout.
- `scripts/run_scenarios.py`: applies explicit scenario shocks to a SARIMAX forecast.
- `data/demo/monthly_exports_synthetic.csv`: synthetic knitwear, woven and total series.
- `config/scenarios.json`: illustrative upside, downside and disruption shocks.
- `notebooks/01_rmg_export_forecasting_walkthrough.ipynb`: interactive modelling walkthrough.
- `tests/`: manual tests for baseline, metrics and scenario calculations.
- `source-code.zip`: downloadable CI-free source package.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_analysis.py
python scripts/run_scenarios.py
```

Open the notebook in JupyterLab, VS Code or Google Colab. Prophet can be added as an optional comparison with `pip install prophet`; the core workflow does not require it.

## Data and limitations

The included monthly export series is synthetic; it is not Bangladesh's observed export history. For empirical work, source monthly statistics from EPB/BGMEA or another documented official dataset and retain the source URL, retrieval date, unit, product coverage, fiscal/calendar basis, revisions and classification notes.

Use chronological holdouts and expanding-window backtesting. Scenario shocks are deterministic stress tests, not causal economic estimates. Historical models cannot independently anticipate trade policy, exchange-rate or geopolitical changes.

No GitHub Actions or CI workflow is configured. Tests are available for manual local execution.

## Licence

MIT. See [LICENSE](LICENSE).
