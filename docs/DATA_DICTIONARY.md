# Data Dictionary

## Monthly export dataset

File: `data/demo/monthly_exports_synthetic.csv`

| Field | Type | Unit | Meaning | Checks |
|---|---|---|---|---|
| month | date | month-start | Observation month | Unique, increasing, one record per month |
| woven_exports_usd_million | numeric | USD million | Woven garment export value | Non-negative; compare against source |
| knit_exports_usd_million | numeric | USD million | Knitwear export value | Non-negative; compare against source |
| total_exports_usd_million | numeric | USD million | Combined RMG exports | Reconcile to components if definition supports it |

The demo is synthetic. For empirical use, the preferred source is EPB monthly export data, with BGMEA's Export Performance page as a convenient compilation. BGMEA states EPB as its source. Preserve the actual downloaded source file.

## Minimum provenance fields

Record:
- publisher and exact source page or document;
- publication/update date and retrieval date;
- fiscal or calendar basis;
- product coverage and classification definition;
- unit, currency, valuation basis and revisions;
- missing months, revisions, reclassifications and transformations;
- licence and reuse conditions.

## Suggested feature table for advanced forecasting

| Feature | Example | Caveat |
|---|---|---|
| Exchange rate | monthly BDT/USD | Confirm whether average, period-end or official rate |
| Global demand | buyer-market imports | Align release date with forecast origin |
| Freight cost | freight index | Avoid look-ahead from revised data |
| Energy cost | industrial tariff | Capture policy effective dates |
| Holiday/calendar | Eid, public holidays, working days | Define in advance and avoid hindsight |
| Policy/disruption indicator | event window | Scenario or causal variable must be clearly distinguished |
