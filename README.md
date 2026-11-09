# Bangladesh RMG Export Forecasting

A monthly time-series analysis framework for Bangladesh's ready-made garment (RMG) exports. The project separates the data preparation problem from the forecasting problem and benchmarks statistical forecasts against a simple seasonal-naive baseline.

> **Evidence status:** the bundled export series is synthetic. Forecast accuracy from this example is not evidence about actual Bangladesh export performance. Replace it with a documented EPB/BGMEA series before making an empirical claim.

## Start here

For a step-by-step explanation, read [the beginner guide](docs/BEGINNER_GUIDE.md). Check the [data dictionary](docs/DATA_DICTIONARY.md) before replacing the synthetic observations.

### Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_analysis.py
python scripts/run_scenarios.py
```

Optional Prophet installation is documented separately in `requirements-prophet.txt`. The core comparison does not depend on Prophet. When Prophet is installed, `scripts/run_analysis.py` adds it to the same chronological holdout comparison.

## What the project does

1. Reads monthly export values with a dated monthly index.
2. Checks that the series is suitable for monthly modelling.
3. Holds out the most recent 12 months in chronological order.
4. Fits a seasonal-naive baseline, Holt-Winters and SARIMAX models; Prophet is included when installed.
5. Compares all models on the same holdout using MAE, RMSE and sMAPE; SARIMAX prediction intervals are exported.
6. Runs expanding-window backtests to see how performance changes across forecast origins, with the same forecast horizon at each origin.
7. Applies explicit upside/downside shocks as what-if scenarios.

The scenario tool applies a percentage adjustment to a forecast. It is a stress-test assumption, not an econometric estimate of the effect of a tariff, exchange-rate change or geopolitical event.

## Understanding the data

The project is designed for monthly export values, normally in USD million. The recommended fields are:

- `month`: month-start date (for example, 2024-07-01).
- `woven_exports_usd_million`: monthly woven garment exports.
- `knit_exports_usd_million`: monthly knitwear exports.
- `total_exports_usd_million`: total RMG exports, ideally reconciled to woven + knit.

Before modelling, document whether a series is based on calendar months or Bangladesh fiscal-year months, whether the values are revised, and whether the product coverage matches the intended definition of RMG exports.

## Forecasting methods

| Method | Role | Strength | Limitation |
|---|---|---|---|
| Seasonal naive | Benchmark | Easy to explain and surprisingly difficult to beat | Repeats last year's seasonal pattern |
| Holt-Winters | Trend + seasonality | Simple interpretable smoothing model | Sensitive to structural breaks and seasonal pattern changes |
| SARIMAX | Autoregressive + differencing + seasonal dynamics | Flexible statistical time-series framework | Requires diagnostics and careful order selection |
| Prophet (optional extension) | Trend and multiple seasonal components | Convenient for business-calendar patterns | Needs validation; defaults are not automatically optimal |

Use the same training period, forecast horizon and evaluation dates for every candidate. Do not randomly shuffle monthly observations.

## Evaluation

- **MAE:** average absolute forecast error, in the same unit as exports (USD million).
- **RMSE:** gives greater penalty to large misses.
- **sMAPE:** percentage-style scale-free comparison, useful where denominators are not near zero.

A model with the lowest score on one holdout is not automatically the best model. Check rolling-origin performance, residual behaviour, intervals, stability and whether the forecast is interpretable enough for the intended decision.

## Bangladesh data sources

- **Export Promotion Bureau (EPB) official export-data portal:** https://epb.gov.bd/pages/epb-export-datas/
- **BGMEA Export Performance:** publishes monthly and annual woven/knit/total apparel export series and identifies EPB as its source: https://www.bgmea.com.bd/page/Export_Performance

When downloading data, capture the retrieval date, reporting period, fiscal-year convention, unit, product classification, revision status and source file. Do not combine fiscal-year totals with monthly values or mix HS-code coverage without documenting the mapping.

## Repository structure

```text
config/       scenario assumptions
data/demo/    synthetic monthly export data
docs/         beginner guide, source notes and model card
notebooks/    interactive analysis walkthrough
scripts/      model comparison and scenario runs
src/          reusable forecasting and evaluation functions
tests/        manual unit tests
outputs/      generated comparison tables and forecasts
```

## Important limitations

- Synthetic data are only used to demonstrate the software.
- A univariate series cannot explain why exports rise or fall.
- Structural breaks, policy changes, order shifts, exchange rates, input costs, logistics and global demand can overwhelm historical patterns.
- Scenarios are deterministic shocks, not causal forecasts.
- No external regressors are included in the default baseline. Add them only when their future values are known or forecastable at the forecast origin.

## Reproducibility

Run the unit tests with `pytest -q`. No GitHub Actions or CI workflow is configured. Notebook execution requires the optional notebook dependencies.

## Licence

MIT for the code. Check the source terms of any data added later.
