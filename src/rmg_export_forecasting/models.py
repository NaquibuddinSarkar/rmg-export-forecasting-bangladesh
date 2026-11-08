"""Transparent monthly RMG export forecasting methods.

The package benchmarks candidate forecasts on the same time window. It does
not automatically identify the cause of an export movement. Scenario shocks
are explicit stress assumptions, not causal estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def validate_monthly_series(series: pd.Series) -> pd.Series:
    """Validate and return a complete, unique month-start indexed numeric series.

    Missing months are not silently dropped or interpolated. A missing month
    is a data issue that the analyst must resolve and document before fitting.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series index must be a pandas DatetimeIndex")
    if series.empty:
        raise ValueError("monthly series cannot be empty")

    values = pd.to_numeric(series, errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("monthly series must contain finite values and no missing months")
    if (values < 0).any():
        raise ValueError("export values cannot be negative")

    month_index = values.index.to_period("M").to_timestamp(how="start")
    if month_index.has_duplicates:
        raise ValueError("there must be exactly one observation per calendar month")
    values.index = month_index
    values = values.sort_index()

    expected = pd.date_range(values.index.min(), values.index.max(), freq="MS")
    missing = expected.difference(values.index)
    if len(missing):
        preview = ", ".join(x.strftime("%Y-%m") for x in missing[:5])
        raise ValueError(f"monthly series has missing months: {preview}")
    if len(values) < 24:
        raise ValueError("at least 24 consecutive monthly observations are required")
    values.index.name = "month"
    values.name = series.name or "export_value"
    return values


def _validate_horizon(horizon: int):
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")


def seasonal_naive(train: pd.Series, horizon: int, season_length: int = 12):
    """Repeat the most recent seasonal pattern for the requested horizon."""
    _validate_horizon(horizon)
    if season_length < 1 or len(train) < season_length:
        raise ValueError("training history must include at least one full season")
    pattern = np.asarray(train.iloc[-season_length:], dtype=float)
    values = np.resize(pattern, horizon)
    index = pd.date_range(
        train.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    return pd.Series(values, index=index, name="seasonal_naive")


def fit_holt_winters(train: pd.Series, horizon: int, season_length: int = 12):
    """Fit additive-trend/additive-seasonal Holt-Winters and forecast forward."""
    _validate_horizon(horizon)
    if len(train) < 2 * season_length:
        raise ValueError("Holt-Winters needs at least two complete seasonal cycles")
    model = ExponentialSmoothing(
        train.astype(float),
        trend="add",
        seasonal="add",
        seasonal_periods=season_length,
        initialization_method="estimated",
    ).fit(optimized=True)
    return model.forecast(horizon)


def fit_sarimax(
    train: pd.Series,
    horizon: int,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
):
    """Fit a SARIMAX model with explicitly supplied non-seasonal/seasonal orders."""
    _validate_horizon(horizon)
    if len(train) < 3 * int(seasonal_order[3]):
        raise ValueError("SARIMAX seasonal fit needs at least three seasonal cycles")
    model = SARIMAX(
        train.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model.get_forecast(steps=horizon).predicted_mean


def fit_sarimax_with_intervals(
    train: pd.Series,
    horizon: int,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    alpha: float = 0.05,
):
    """Return SARIMAX mean forecast and model-based prediction interval."""
    _validate_horizon(horizon)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if len(train) < 3 * int(seasonal_order[3]):
        raise ValueError("SARIMAX seasonal fit needs at least three seasonal cycles")
    fitted = SARIMAX(
        train.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    prediction = fitted.get_forecast(steps=horizon)
    mean = prediction.predicted_mean.rename("forecast")
    interval = prediction.conf_int(alpha=alpha)
    interval.columns = ["lower", "upper"]
    result = pd.concat([mean, interval], axis=1)
    return result, fitted


def score_forecast(actual: pd.Series, predicted: pd.Series):
    """Compare a forecast with actual values over their common timestamps.

    MAE/RMSE use the original series unit. sMAPE is reported in percent and
    remains finite when either actual or predicted values are zero.
    """
    joined = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")],
        axis=1,
        join="inner",
    ).dropna()
    if joined.empty:
        raise ValueError("actual and predicted series have no aligned observations")
    a = joined["actual"].to_numpy(dtype=float)
    p = joined["predicted"].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("actual and predicted values must be finite")
    denominator = np.abs(a) + np.abs(p)
    smape_values = np.divide(
        2.0 * np.abs(a - p),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "n": int(len(joined)),
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "sMAPE_pct": float(100.0 * np.mean(smape_values)),
    }


def expanding_window_backtest(
    series: pd.Series,
    min_train: int = 36,
    horizon: int = 12,
    step: int | None = None,
):
    """Evaluate three models across chronological expanding-window origins.

    At each origin, only the observations before that origin are used for
    fitting. The default step equals the horizon, so test windows do not
    overlap. The result contains one row per model and forecast origin.
    """
    values = validate_monthly_series(series)
    _validate_horizon(horizon)
    if min_train < 3 * 12 or min_train >= len(values):
        raise ValueError("min_train must be at least 36 months for the default yearly SARIMAX and smaller than series length")
    step = horizon if step is None else int(step)
    if step < 1:
        raise ValueError("step must be positive")

    rows = []
    for end in range(min_train, len(values) - horizon + 1, step):
        train = values.iloc[:end]
        test = values.iloc[end : end + horizon]
        candidates = {
            "Seasonal naive": seasonal_naive(train, horizon),
            "Holt-Winters": fit_holt_winters(train, horizon),
            "SARIMAX": fit_sarimax(train, horizon),
        }
        for model_name, prediction in candidates.items():
            prediction.index = test.index
            rows.append({
                "origin": train.index[-1].strftime("%Y-%m"),
                "model": model_name,
                "forecast_horizon_months": int(horizon),
                **score_forecast(test, prediction),
            })
    if not rows:
        raise ValueError("not enough observations for the requested backtest setup")
    return pd.DataFrame(rows)


def scenario_adjustment(forecast: pd.Series, percentage_shock: float):
    """Apply a transparent constant percentage shock to a forecast path.

    This creates a deterministic what-if case. It does not estimate the
    causal effect of a tariff, exchange rate, policy change or disruption.
    """
    if not np.isfinite(percentage_shock) or percentage_shock <= -100:
        raise ValueError("percentage_shock must be finite and greater than -100")
    values = pd.to_numeric(forecast, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("forecast values must be finite")
    return values * (1.0 + percentage_shock / 100.0)
