"""Monthly RMG export forecasting baselines and statistical models."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def validate_monthly_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float).dropna().sort_index()
    if len(s) < 30:
        raise ValueError("at least 30 monthly observations are required")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("series index must be a DatetimeIndex")
    if not s.index.is_unique:
        raise ValueError("monthly index must be unique")
    return s.asfreq("MS")


def seasonal_naive(train: pd.Series, horizon: int, season_length: int = 12):
    if horizon < 1 or len(train) < season_length:
        raise ValueError("positive horizon and sufficient history are required")
    pattern = np.asarray(train.iloc[-season_length:], dtype=float)
    values = np.resize(pattern, horizon)
    index = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series(values, index=index, name="seasonal_naive")


def fit_holt_winters(train: pd.Series, horizon: int, season_length: int = 12):
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=season_length,
        initialization_method="estimated"
    ).fit(optimized=True)
    return model.forecast(horizon)


def fit_sarimax(train: pd.Series, horizon: int,
                order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    return model.get_forecast(steps=horizon).predicted_mean


def score_forecast(actual: pd.Series, predicted: pd.Series):
    joined = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    return {
        "MAE": float(mean_absolute_error(joined.actual, joined.predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(joined.actual, joined.predicted)))
    }


def expanding_window_backtest(series: pd.Series, min_train: int = 36, horizon: int = 12):
    """One-origin-at-a-time expanding evaluation for a seasonal-naive benchmark."""
    s = validate_monthly_series(series)
    rows = []
    for end in range(min_train, len(s) - horizon + 1, horizon):
        train, test = s.iloc[:end], s.iloc[end:end + horizon]
        pred = seasonal_naive(train, horizon)
        pred.index = test.index
        metrics = score_forecast(test, pred)
        rows.append({"origin": str(train.index[-1].date()), **metrics})
    return pd.DataFrame(rows)


def scenario_adjustment(forecast: pd.Series, percentage_shock: float):
    """Apply an explicit constant scenario shock; this is not a causal estimate."""
    return forecast * (1.0 + percentage_shock / 100.0)
