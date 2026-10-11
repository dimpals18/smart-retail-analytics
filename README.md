# 📊 Smart Retail Analytics

An end-to-end data analytics project: synthetic retail data → collection → cleaning → SQL database → analysis → forecasting → interactive dashboard.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py                     # runs the whole pipeline
python -m streamlit run dashboard.py  # opens the dashboard
```
(Or `python run_all.py --dashboard` to do both.) Run commands from the project root folder.

## Pipeline

| Step | Script | What it does | Output (`data/`) |
|---|---|---|---|
| 0 | `create_data.py` | Generates 200 customers, 30 products, ~3,600 orders, 300 reviews (3 years of history, trend + seasonality, deliberately messy) | `customers.csv`, `products.json/.xml`, `orders.csv`, `reviews.txt` |
| 1 | `src/data_collection.py` | Reads CSV / JSON / XML / TXT, cross-checks JSON vs XML, joins into one table | `raw_orders.csv`, `reviews.csv` |
| 2 | `src/data_cleaning.py` | Removes duplicates and invalid quantities, standardises city names, fixes impossible ages, imputes missing sales, corrects outliers | `cleaned_orders.csv`, `cleaning_report.json` |
| 3 | `src/database.py` | Loads normalised tables (customers, products, orders) into SQLite | `retail.db` |
| 4 | `src/analysis.py` | SQL analysis, RFM customer segmentation, Pareto (80/20) product analysis | `rfm_segments.csv`, `product_pareto.csv`, `category_summary.csv` |
| 5 | `src/forecasting.py` | Compares 3 models on a 6-month hold-out, forecasts the next 3 months with the best one | `monthly_sales.csv`, `model_metrics.csv`, `sales_forecast.csv` |
| 6 | `dashboard.py` | Streamlit dashboard with filters, KPIs, charts, forecast and reports | – |

## Data cleaning

| Problem injected | Fix |
|---|---|
| 40 duplicate orders | Dropped by `order_id` |
| 6 zero / negative quantities | Rows removed |
| City spelled `nagpur`, `NAGPUR`, ` Nagpur ` | Trimmed and title-cased |
| Missing or impossible ages (0, 120, 150) | Replaced with the median age |
| 15 missing sales values | Imputed as price × quantity |
| 8 sales values 10× too large | Corrected to price × quantity |

## Forecasting models

* **Moving average (3 months)** – baseline; flat forecast of the recent average.
* **Exponential smoothing** – weights recent months more heavily.
* **Holt-Winters** – captures trend and yearly seasonality.

Models are scored with MAE, RMSE and MAPE on the last 6 months, which are held out of training. The lowest-MAPE model is refitted on all data to forecast.

## Dashboard features

Filters (city, category, date range) • KPI cards with change vs previous period • Monthly trend with moving average • Category / city / product charts • City × category heatmap • Age-group analysis • RFM segments • Pareto chart • Forecast with model comparison • Review sentiment • Cleaning report • CSV downloads.

## Tech stack
Python, Pandas, NumPy, SQLAlchemy (SQLite), Statsmodels, Plotly, Streamlit.
