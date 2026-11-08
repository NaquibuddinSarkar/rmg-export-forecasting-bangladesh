# Forecast model card

## Intended use
Exploratory analysis and method comparison for monthly apparel-export series.

## Candidate methods
Seasonal-naive baseline, Holt-Winters additive smoothing and SARIMAX. Prophet may be added as an optional dependency and benchmark.

## Evaluation
Use chronological holdouts and expanding-window backtesting. Report MAE and RMSE on a common forecast horizon; MAPE is unsuitable when actual values can approach zero. Inspect residual autocorrelation and forecast intervals.

## Limitations
Forecasts are conditional on historical patterns and cannot anticipate policy changes, trade restrictions, exchange-rate shocks or geopolitical disruptions without explicit scenario inputs. The bundled data are synthetic.
