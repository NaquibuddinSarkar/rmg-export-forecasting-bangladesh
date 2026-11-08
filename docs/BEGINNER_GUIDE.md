# Beginner Guide: Bangladesh RMG Export Forecasting

## 1. What are we forecasting?

This project studies monthly exports of Bangladesh's ready-made garments (RMG). It is designed to forecast a monthly value series and compare a few standard time-series methods.

The default data are synthetic. The sample output demonstrates the code workflow only; it does not describe actual Bangladesh exports.

## 2. Understand the columns

The preferred data structure contains one record per month:
- month: a monthly date, normally stored as the first day of the month;
- woven_exports_usd_million: woven export value;
- knit_exports_usd_million: knitwear export value;
- total_exports_usd_million: total RMG value.

Check whether the source reports calendar months or fiscal-year months. Bangladesh government statistics commonly present July–June financial-year groupings in addition to monthly series. Preserve the source's convention.

## 3. Why time series need chronological splits

A forecast uses the past to predict the future. A random train/test split can accidentally let future observations influence model evaluation. This project reserves the latest 12 months as a holdout and uses earlier months for fitting.

For a stronger analysis, evaluate several forecast origins through rolling or expanding-window backtesting.

## 4. Models in everyday language

**Seasonal naive:** predicts each month using the value from the same month one year earlier. It is a simple benchmark.

**Holt-Winters:** smooths the level, trend and repeating seasonal pattern. It is often a useful business baseline.

**SARIMAX:** models relationships with past values and errors, with differencing and seasonal terms. It can be extended with external variables, but future values of those variables must be available or forecasted.

**Prophet (optional):** a trend/seasonality model that can include calendar effects. It should be compared against simple benchmarks rather than assumed superior.

## 5. Run the analysis

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_analysis.py
python scripts/run_scenarios.py
```

The first command writes model-comparison metrics to `outputs/model_comparison_demo.csv`. The second applies preconfigured percentage shocks to a forecast.

## 6. How to read the metrics

- MAE: average absolute error, in USD million.
- RMSE: root mean squared error; large misses have more influence.
- MAPE/sMAPE: percentage-style errors; avoid ordinary MAPE when actual values can be zero or very small.

Always compare models on the same dates and horizon. Do not compare the MAE of a one-month forecast with the MAE of a twelve-month forecast.

## 7. What makes a credible Bangladesh RMG study?

Use a documented EPB or BGMEA monthly series. Record the retrieval date, unit, period, revisions and product definition. Check the total equals woven plus knitwear if those series are intended to be exhaustive. Avoid mixing monthly data with fiscal-year cumulative totals.

Add context only when the data support it: exchange rates, order book indicators, global demand, freight costs, energy cost, policy changes, labour conditions and major disruptions. Explain whether a variable is a predictor, a scenario assumption or only background context.

## 8. Common mistakes

- Randomly splitting chronological data.
- Fitting the model after using the holdout to choose model orders.
- Applying a constant percentage scenario and describing it as a causal policy impact.
- Mixing fiscal and calendar years.
- Ignoring data revisions or definitions.
- Selecting a model solely because it wins on one synthetic holdout.

## 9. Recommended extensions

Use rolling-origin comparison for all candidate models, add prediction intervals, inspect residual ACF, evaluate knit and woven separately, test an optional Prophet model, and investigate whether external regressors improve out-of-sample forecasts.
