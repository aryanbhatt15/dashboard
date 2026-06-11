# CC March (H) Contract — Analysis Dashboard

## Setup
```bash
pip install dash plotly pandas openpyxl numpy scipy
```

## Run
```bash
python app.py
```
Then open http://127.0.0.1:8050

## Files
- `app.py`           — Dash dashboard (6 tabs)
- `data_engine.py`   — All data loading, period logic, analysis functions
- `*.xlsx`           — Source data (place in same folder as scripts)

## Period Definitions (CC March)
| Period | Start | End | Approx BDs |
|--------|-------|-----|-----------|
| P1 | Day after CC Dec LTD | Day before Goldman roll start | 33–38 |
| P2 | 5th BD of February | 9th BD of February | 5 |
| P3 | Day after Goldman roll end | CC March LTD | 20–25 |

Goldman roll = 5th to 9th business day of the month preceding expiry.
For March = February. For other contracts change `roll_month` in `get_boundaries()`.

## Tabs
1. **Overview** — Period timeline, boundaries table per year, OI cliff analysis
2. **Outright** — OHLC/OI/Vol chart, normalised overlay, range by period, heatmap
3. **H-K Spread** — Spread single-year + overlay + hit-rate bar + direction heatmap
4. **HKN Fly** — Fly single-year + overlay + hit-rate bar + heatmap
5. **Roll Window** — Deep dive into P2: avg move per roll day, OI/Vol, scatter
6. **Pattern Engine** — Auto-detected ≥65% hit-rate BD patterns, bias heatmap, scorecard

## Extending to Other Contracts
In `data_engine.py`, `get_boundaries()` currently handles March (H).
To add other contracts, replace `cc_dec_ltd(yr-1)` / `cc_mar_ltd(yr)` with
the appropriate prev-contract LTD and current contract LTD functions,
and change `roll_month = jul_ltd.month - 1` accordingly.
