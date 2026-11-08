import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rmg_export_forecasting.models import seasonal_naive, score_forecast, scenario_adjustment


def monthly_series(n=60):
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    return pd.Series(np.arange(n, dtype=float) + 10*np.sin(2*np.pi*np.arange(n)/12), index=idx)


def test_seasonal_naive_repeats_last_year_pattern():
    s = monthly_series()
    pred = seasonal_naive(s.iloc[:-12], 12)
    np.testing.assert_allclose(pred.values, s.iloc[-24:-12].values)


def test_score_forecast():
    s = pd.Series([1., 2., 3.])
    result = score_forecast(s, pd.Series([1., 3., 2.]))
    assert result["MAE"] == 2/3
    assert result["RMSE"] > 0


def test_scenario_adjustment():
    s = pd.Series([100., 200.])
    np.testing.assert_allclose(scenario_adjustment(s, -10).values, [90, 180])
