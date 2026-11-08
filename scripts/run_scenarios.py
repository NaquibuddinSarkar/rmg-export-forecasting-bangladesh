"""Apply simple deterministic scenario shocks to a baseline forecast."""
from pathlib import Path
import sys
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rmg_export_forecasting.models import fit_sarimax, scenario_adjustment

def main():
    df=pd.read_csv(ROOT/"data/demo/monthly_exports_synthetic.csv",parse_dates=["month"])
    series=df.set_index("month")["total_exports_usd_million"].asfreq("MS")
    forecast=fit_sarimax(series,12)
    scenarios=json.loads((ROOT/"config/scenarios.json").read_text())
    result=pd.DataFrame({"baseline":forecast})
    for key,setting in scenarios.items():
        if isinstance(setting,dict) and "percentage_shock" in setting:
            result[key]=scenario_adjustment(forecast,setting["percentage_shock"])
    out=ROOT/"outputs"; out.mkdir(exist_ok=True)
    result.to_csv(out/"scenario_forecasts_demo.csv")
    print(result.round(2))

if __name__=="__main__":
    main()
