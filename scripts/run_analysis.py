"""Run the synthetic RMG export forecast, holdout comparison and backtest."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmg_export_forecasting.models import (
    validate_monthly_series,
    seasonal_naive,
    fit_holt_winters,
    fit_sarimax_with_intervals,
    fit_prophet,
    score_forecast,
    expanding_window_backtest,\n    ljung_box_diagnostics,
)


def main():
    demo_path = ROOT / "data" / "demo" / "monthly_exports_synthetic.csv"
    frame = pd.read_csv(demo_path, parse_dates=["month"])
    series = validate_monthly_series(
        frame.set_index("month")["total_exports_usd_million"]
    )

    # Keep the final year untouched until the candidate models are fitted.
    horizon = 12
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    forecasts = {
        "Seasonal naive": seasonal_naive(train, horizon),
        "Holt-Winters": fit_holt_winters(train, horizon),
    }
    sarimax_interval, sarimax_fitted = fit_sarimax_with_intervals(train, horizon, alpha=0.05)
    forecasts["SARIMAX"] = sarimax_interval["forecast"]

    try:
        forecasts["Prophet"] = fit_prophet(train, horizon)
        print("Prophet comparison included.")
    except ImportError:
        print("Prophet not installed; skipping optional Prophet comparison. "
              "Install requirements-prophet.txt to enable it.")

    metric_rows = []
    forecast_table = pd.DataFrame({"actual": test})
    for name, prediction in forecasts.items():
        prediction.index = test.index
        forecast_table[name] = prediction
        metric_rows.append({"model": name, **score_forecast(test, prediction)})

    comparison = pd.DataFrame(metric_rows).sort_values("RMSE")
    backtest = expanding_window_backtest(\n        series, min_train=36, horizon=12, step=12\n    )\n    residual_diagnostics = ljung_box_diagnostics(sarimax_fitted.resid, lags=12)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison_demo.csv", index=False)
    forecast_table.to_csv(output_dir / "holdout_forecasts_demo.csv")
    sarimax_interval.to_csv(output_dir / "sarimax_prediction_intervals_demo.csv")
    backtest.to_csv(output_dir / "expanding_window_backtest_demo.csv", index=False)\n    residual_diagnostics.to_csv(output_dir / "sarimax_ljung_box_demo.csv")

    print("MODEL COMPARISON — SYNTHETIC DEMONSTRATION")
    print(comparison.to_string(index=False))
    print("\nSAMPLE HOLDOUT FORECASTS")
    print(forecast_table.head().to_string())
    print("\nEXPANDING-WINDOW BACKTEST")
    print(backtest.groupby("model")[["MAE", "RMSE", "sMAPE_pct"]].mean().to_string())
    print(
        "\nWARNING: The included monthly export series is synthetic. "
        "This run demonstrates methodology and software only, not actual "
        "Bangladesh RMG export performance."
    )


if __name__ == "__main__":
    main()
