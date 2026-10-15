# Executive summary: Retail demand forecasting

**Prepared by: Anurag Moyal**

**Scope.** 913,000 daily sales records, 10 stores × 50 items, 2013–2017. Goal: forecast the next 90 days so stock can be ordered ahead of demand.
All numbers are from rolling-origin backtests (the forecast is always made from data available at the time, then compared with what actually happened).

## What the data looks like
* Demand grows every year (+15 %, +4 %, +9 %, +4 % for 2014–2017) with a strong **weekly cycle** (Sunday about 1.18× vs Monday 0.79× of average) and a strong **yearly cycle** (July peak 1.28×, January trough 0.70×).
* The seasonal shape is identical across stores and items; only the volume level differs (best store sells 1.8× the smallest; best item 4.8× the smallest).
* No missing days, duplicates or negative sales.

## Which forecast to use depends on the level of decision

| Level | Best approach | Error (WAPE) | Why |
|---|---|---|---|
| Company total / store | "Same weekday last year × recent growth" rule | 1.1 % total, 2.3 % per store | Yearly pattern is so stable that a simple rule is already near the noise floor; fitted models add estimation noise |
| **Store-item (what replenishment needs)** | **One global LightGBM model** | **10.0 %** vs 14.1 % for the year-over-year rule, 19.7 % for naive | Item-level data is noisy; a model that learns from all 500 series at once handles it far better |

* At total level, LightGBM scored MAPE 3.3 % against 2.2 % for the simple year-over-year rule; Prophet 3.9 %, SARIMAX 4.4 %, ETS 12.3 %, naive 17.3 %.
* ETS, which sees only weekly seasonality, performs poorly because it misses the yearly cycle; SARIMAX works once annual Fourier terms are added.
* The global LightGBM beat the year-over-year rule on every one of the 500 series in the final test window.

## Business impact
Using the LightGBM forecast instead of the year-over-year rule cuts the required **safety stock by about 29 %** (and by about 48 % versus a naive forecast) at a 95 % service level and 7-day lead time. This is an
approximation: it assumes forecast errors are independent from day to day.

## Deliverable
`reports/forecast_2018Q1_store_item.csv` holds the daily forecast per store and item for 1 January to 31 March 2018 (45,000 rows); the total is about 2.23 million units.

## Recommendations
1. Use the simple year-over-year rule for aggregate planning (budgets, store totals); it is transparent and as accurate as anything fitted.
2. Use the global LightGBM model for store-item replenishment, and retrain it monthly.
3. Track bias (over- vs under-forecasting) and WAPE per store and volume class, and compare against the simple rule as a standing benchmark.
4. Add promotions, prices and a holiday calendar if available; these are not in the competition data and would be the next accuracy gains.

## Caveats
* The competition data is unusually regular (identical seasonality everywhere), which favours simple seasonal rules; real retail data is messier, so re-validate before relying on the margin.
* Prophet and SARIMAX were not exhaustively tuned.
* The 2018 forecast cannot be checked yet; its accuracy is expected to resemble the backtest.
