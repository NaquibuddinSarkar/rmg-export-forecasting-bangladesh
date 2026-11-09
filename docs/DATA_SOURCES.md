# Bangladesh RMG export data sources and handling notes

The forecasting code is designed for a continuous monthly series. The demo observations are synthetic. Replace them with a downloaded and documented official series before interpreting model performance as evidence about Bangladesh exports.

## Primary sources

| Source | What it provides | Data-handling notes |
|---|---|---|
| [Export Promotion Bureau (EPB) export data portal](https://epb.gov.bd/pages/epb-export-datas/) | Official export-data downloads and reporting tables | Record the exact download, series definition, revision date, reporting period and unit |
| [BGMEA Export Performance](https://www.bgmea.com.bd/page/Export_Performance) | Woven, knitwear and total RMG export performance tables; BGMEA identifies EPB as its source | Check the page's month/fiscal-year basis and reconcile component totals before modelling |

## Recommended data fields

- month (month-start date);
- knit exports (USD million);
- woven exports (USD million);
- total exports (USD million).

If monthly totals are constructed as knit + woven, document that definition. If official data contain additional apparel categories or a different product classification, do not silently assume the two series are exhaustive.

## Data quality checklist

1. There is exactly one row per calendar month.
2. No months are missing. Do not silently interpolate missing observations.
3. Values are numeric, finite and non-negative.
4. The currency and unit are consistent across the time series.
5. Woven + knit equals total only if the source defines total that way.
6. Fiscal-year aggregates are not mixed with monthly values.
7. Revisions and classification changes are recorded.
8. The holdout period and forecast horizon are documented.

## Suggested external variables for later research

Potential explanatory features include exchange rates, buyer-market imports, energy prices, freight indicators, working days, holidays and policy/disruption dates. Their actual values must be available at the historical forecast origin; using future realised predictor values causes look-ahead leakage.

## Reproducibility record

Store the original source file unchanged outside the derived-data folder. Record the source title, publisher, URL, date accessed, reporting period, unit, classification, licence and all transformations in a dataset-specific provenance note.
