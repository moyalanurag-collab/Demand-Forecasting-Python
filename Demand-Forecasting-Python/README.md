# Retail Demand Forecasting: Baselines vs ARIMA vs ETS vs Prophet vs Machine Learning

**Author: Anurag Moyal** · Data Analytics Portfolio Project

Forecasting the next 90 days of retail demand for **10 stores × 50 items (500 daily series, 913,000 observations, 2013–2017)**, comparing
simple baselines, classical statistics (ETS, SARIMAX), Prophet and a global gradient-boosting model, with **rolling-origin backtesting**, proper error
metrics (MAPE, RMSE, WAPE, MASE) and a business translation (safety stock).

| Notebook | What it does |
|---|---|
| `notebooks/01_eda_and_data_quality.ipynb` | Data audit, trend, weekly and yearly seasonality, STL decomposition, stationarity tests (ADF/KPSS), ACF/PACF, ABC analysis |
| `notebooks/02_total_demand_backtesting.ipynb` | 7 models on total daily demand, 4 rolling 90-day folds, Diebold-Mariano test, residual diagnostics |
| `notebooks/03_store_item_forecasting.ipynb` | 500 series: ETS per series vs one global LightGBM (direct multi-horizon), 3 folds, accuracy by level / store / volume class / horizon, feature importance, safety-stock impact, **production forecast for Jan–Mar 2018** |

Executive write-up: `reports/Executive_Summary.md`. Metrics tables: `reports/results_*.csv`. Final forecast file: `reports/forecast_2018Q1_store_item.csv` (45,000 rows).

## Headline results (rolling-origin backtests, 90-day horizon)
**Total demand (1 series, 4 folds)**

| Model | MAPE | RMSE (units/day) |
|---|---|---|
| Seasonal naive YoY × growth (business rule) | **2.2 %** | **914** |
| LightGBM (lag features) | 3.3 % | 1,066 |
| Prophet | 3.9 % | 1,374 |
| SARIMAX (weekly + annual Fourier terms) | 4.4 % | 1,511 |
| ETS (Holt-Winters, weekly) | 12.3 % | 4,484 |
| Seasonal naive (repeat last week) | 12.8 % | 4,718 |
| Naive (28-day mean) | 17.3 % | 6,074 |

**Store-item level (500 series, 3 folds)**

| Model | WAPE | MAPE | RMSE |
|---|---|---|---|
| **Global LightGBM** | **10.0 %** | **12.1 %** | **8.15** |
| Seasonal naive YoY × growth | 14.1 % | 16.7 % | 11.51 |
| ETS per series | 16.1 % | 18.6 % | 13.18 |
| Naive (28-day mean) | 19.7 % | 22.2 % | 16.32 |

**The honest finding:** the best model depends on the level. At total and store level a well-built *"same weekday last year × recent growth"* rule is as good as
or better than any fitted model (demand has a very stable yearly shape). At the noisy store-item level the global LightGBM model cuts error by ~29 % versus that rule
and ~49 % versus naive, which translates to about **29 % less safety stock** for the same service level. Fitted models also add value when seasonality is *not* this stable.

## Project structure
```
data/raw/train.csv                913,000 daily rows (public copy of Kaggle "Store Item Demand Forecasting Challenge")
src/forecast_utils.py             metrics, backtest folds, leak-free feature builder
src/notebook_sources/*.py         notebook sources (percent format)
src/build_notebooks.py            converts and executes the notebooks
notebooks/                        executed notebooks (outputs included)
reports/                          executive summary, result tables, forecast file, 16 figures
```

## Reproduce
```bash
pip install -r requirements.txt
python -m ipykernel install --user --name python3
python src/build_notebooks.py      # runs all three notebooks, about 10-15 minutes
```
Or double-click `run_all.bat` on Windows.

## Method notes
* **No leakage:** every model gets only data up to the forecast origin; LightGBM training rows are cut at the origin; features are built from the last observed day.
* **Direct multi-horizon LightGBM:** one global model predicts day 1…90 ahead using lag features, rolling means, same-weekday means, last-year values, growth ratio and calendar fields (trained on 2.1 million rows).
* **Metrics:** MAPE and RMSE as requested, plus WAPE (volume-weighted, robust to tiny items), MAE, bias and MASE. MASE is scaled by the in-sample weekly-naive error.
* **Statistical comparison:** a Diebold-Mariano test on pooled errors (approximate Newey-West variance).
* **Safety stock:** z × σ(error) × √lead-time, with 95 % service, 7-day lead time and σ from the backtests; it assumes independent daily errors, so treat it as indicative.

## Limitations
* The competition data is very regular (identical seasonal shape across all series), which favours simple seasonal rules; real retail data is usually messier (promotions, stock-outs, holidays).
* No holiday calendar, price or promotion information is available in the data.
* Prophet and SARIMAX use sensible fixed settings, not exhaustive tuning (SARIMAX order was not selected by search).
* The 2018 forecast cannot be verified; its uncertainty band uses the backtest RMSE of the total forecast.

## CV bullets (edit to taste)
* *Built a demand-forecasting pipeline on 913k daily retail observations (500 store-item series) comparing naive, ETS, SARIMAX, Prophet and LightGBM with rolling-origin backtesting; a global LightGBM model cut item-level WAPE from 14.1 % to 10.0 % versus a seasonal year-over-year benchmark and 19.7 % for naive.*
* *Showed that a simple year-over-year growth rule beats ML at aggregate level (MAPE 2.2 % vs 3.3 %) while ML wins at item level, and translated the accuracy gain into a ~29 % safety-stock reduction.*

## Data source
"Store Item Demand Forecasting Challenge" (Kaggle competition `demand-forecasting-kernels-only`); this copy comes from the public GitHub repository
`allmeidaapedro/Store-Item-Demand-Forecasting` (`input/train.csv`; 913,000 rows, 10 stores, 50 items, 2013-01-01 to 2017-12-31).
