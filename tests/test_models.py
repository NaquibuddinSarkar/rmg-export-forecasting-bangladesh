import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmg_export_forecasting.models import (
    validate_monthly_series,
    seasonal_naive,
    score_forecast,
    scenario_adjustment,
)


def monthly_series(n=60):
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    values = np.arange(n, dtype=float) + 10 * np.sin(2 * np.pi * np.arange(n) / 12)
    return pd.Series(values, index=idx, name="exports")


def test_validate_monthly_series_normalises_dates():
    s = monthly_series()
    s.index = s.index + pd.Timedelta(days=14)
    cleaned = validate_monthly_series(s)
    assert cleaned.index[0] == pd.Timestamp("2018-01-01")
    assert cleaned.index.is_monotonic_increasing
    assert len(cleaned) == 60


def test_validate_monthly_series_rejects_missing_month():
    s = monthly_series().drop(pd.Timestamp("2019-05-01"))
    with pytest.raises(ValueError, match="missing months"):
        validate_monthly_series(s)


def test_validate_monthly_series_rejects_negative_values():
    s = monthly_series()
    s.iloc[5] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_monthly_series(s)


def test_seasonal_naive_repeats_last_year_pattern():
    s = monthly_series()
    pred = seasonal_naive(s.iloc[:-12], 12)
    np.testing.assert_allclose(pred.values, s.iloc[-24:-12].values)


def test_score_forecast_includes_smape_and_is_finite_with_zero():
    actual = pd.Series([0.0, 2.0, 3.0])
    predicted = pd.Series([0.0, 3.0, 2.0])
    result = score_forecast(actual, predicted)
    assert result["n"] == 3
    assert result["MAE"] == pytest.approx(2 / 3)
    assert result["RMSE"] > 0
    assert np.isfinite(result["sMAPE_pct"])


def test_scenario_adjustment():
    s = pd.Series([100.0, 200.0])
    np.testing.assert_allclose(scenario_adjustment(s, -10).values, [90, 180])


def test_scenario_rejects_loss_of_all_value():
    with pytest.raises(ValueError):
        scenario_adjustment(pd.Series([100.0]), -100)


def test_invalid_backtest_configuration_rejected():
    from rmg_export_forecasting.models import expanding_window_backtest
    with pytest.raises(ValueError, match="at least 36"):
        expanding_window_backtest(monthly_series(), min_train=24, horizon=12)
