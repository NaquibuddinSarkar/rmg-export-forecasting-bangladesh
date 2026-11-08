# RMG Export Forecasting — Bangladesh

A time-series analysis and forecasting framework for Bangladesh ready-made garment (RMG) export trends. The complete project source package is provided in [source-code.zip](./source-code.zip).

## What is included

The source archive contains the Python forecasting package, a Jupyter walkthrough, synthetic monthly export data, seasonal-naive and statistical model benchmarks, expanding-window backtesting, scenario analysis, tests and source documentation.

Methods include seasonal-naive, Holt-Winters and SARIMAX; Prophet is an optional dependency.

## Quick start

Download and extract [source-code.zip](./source-code.zip), then run from the extracted project directory:

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

For optional Prophet support, install the package listed in `requirements-prophet.txt`.

## Data status

The included monthly export series is synthetic and does not represent Bangladesh's actual export statistics. The project includes a candidate-source register; replace the demonstration series with data from EPB, BGMEA or other documented official sources, recording definitions, units, release dates and revisions.

**No GitHub Actions or CI workflow is included, as requested.** Local tests are retained for manual verification.

[Download complete source package](./source-code.zip)
