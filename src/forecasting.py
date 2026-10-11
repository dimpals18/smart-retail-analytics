"""
Step 5 - Sales forecasting.

Three models are compared on a hold-out set (the last 6 months):
    * Moving average (3-month, flat forecast)
    * Simple exponential smoothing
    * Holt-Winters (trend + yearly seasonality)
The best model by MAPE is refitted on all data and forecasts the next 3 months.

Outputs (in data/): monthly_sales.csv, model_metrics.csv, sales_forecast.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

DATA = Path("data")
HOLDOUT = 6
HORIZON = 3


def forecast_moving_average(train: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(train.tail(3).mean(), steps)


def forecast_ses(train: pd.Series, steps: int) -> np.ndarray:
    fit = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    return fit.forecast(steps).to_numpy()


def forecast_holt_winters(train: pd.Series, steps: int) -> np.ndarray:
    fit = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated"
    ).fit()
    return fit.forecast(steps).to_numpy()


MODELS = {
    "Moving Average (3m)": forecast_moving_average,
    "Exponential Smoothing": forecast_ses,
    "Holt-Winters": forecast_holt_winters,
}


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err / actual)) * 100),
    }


def main() -> None:
    orders = pd.read_csv(DATA / "cleaned_orders.csv", parse_dates=["order_date"])
    monthly = orders.set_index("order_date")["sales"].resample("MS").sum().rename("sales").to_frame()
    series = monthly["sales"]

    # Chart-friendly columns used by the dashboard
    monthly["moving_average"] = series.rolling(3).mean()
    monthly["exponential_smoothing"] = SimpleExpSmoothing(series, initialization_method="estimated").fit().fittedvalues
    monthly.rename_axis("date").to_csv(DATA / "monthly_sales.csv")

    # Hold-out evaluation
    train, test = series.iloc[:-HOLDOUT], series.iloc[-HOLDOUT:]
    rows = []
    for name, func in MODELS.items():
        rows.append({"model": name, **evaluate(test.to_numpy(), func(train, HOLDOUT))})
    metrics = pd.DataFrame(rows).sort_values("MAPE_%").reset_index(drop=True)
    metrics.round(2).to_csv(DATA / "model_metrics.csv", index=False)

    # Refit best model on all data and forecast
    best = metrics.iloc[0]["model"]
    values = MODELS[best](series, HORIZON)
    future = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    forecast = pd.DataFrame({"date": future, "forecast_sales": values.round(2), "model": best})
    forecast.to_csv(DATA / "sales_forecast.csv", index=False)

    print("Forecasting complete")
    print(metrics.round(2).to_string(index=False))
    print(f"\nBest model: {best}")
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
