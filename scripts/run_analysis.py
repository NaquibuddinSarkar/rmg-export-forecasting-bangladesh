"""Run demonstration RMG export forecasts; inputs are synthetic."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rmg_export_forecasting.models import (
    seasonal_naive, fit_holt_winters, fit_sarimax, score_forecast
)


def main():
    df = pd.read_csv(ROOT / "data" / "demo" / "monthly_exports_synthetic.csv", parse_dates=["month"])
    series = df.set_index("month")["total_exports_usd_million"].asfreq("MS")
    horizon = 12
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    forecasts = {
        "Seasonal naive": seasonal_naive(train, horizon),
        "Holt-Winters": fit_holt_winters(train, horizon),
        "SARIMAX": fit_sarimax(train, horizon)
    }
    rows = []
    for name, pred in forecasts.items():
        pred.index = test.index
        rows.append({"model": name, **score_forecast(test, pred)})
    output = pd.DataFrame(rows).sort_values("RMSE")
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output.to_csv(output_dir / "model_comparison_demo.csv", index=False)
    print(output.to_string(index=False))
    print("\nWARNING: Included export observations are synthetic; replace with sourced EPB/BGMEA data.")


if __name__ == "__main__":
    main()
