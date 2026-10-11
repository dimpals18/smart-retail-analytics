"""
Step 2 - Data cleaning.

Every fix is counted and written to data/cleaning_report.json so the
before/after story is easy to show in a report or viva.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")


def main() -> None:
    df = pd.read_csv(DATA / "raw_orders.csv")
    report = {"rows_raw": int(len(df))}

    # 1. Dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    report["invalid_dates_removed"] = int(df["order_date"].isna().sum())
    df = df.dropna(subset=["order_date"])

    # 2. Duplicate orders
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    report["duplicate_orders_removed"] = int(before - len(df))

    # 3. Invalid quantities (zero / negative / missing)
    bad_qty = df["quantity"].isna() | (df["quantity"] <= 0)
    report["invalid_quantity_removed"] = int(bad_qty.sum())
    df = df[~bad_qty].copy()

    # 4. Inconsistent city names ("nagpur", "NAGPUR", " Nagpur ")
    raw_cities = df["city"].astype(str)
    df["city"] = raw_cities.str.strip().str.title()
    report["city_rows_standardised"] = int((raw_cities != df["city"]).sum())

    # 5. Age: impossible values -> missing -> median
    invalid_age = (df["age"] < 18) | (df["age"] > 65)
    df.loc[invalid_age, "age"] = np.nan
    median_age = df.drop_duplicates("customer_id")["age"].median()
    report["age_rows_filled"] = int(df["age"].isna().sum())
    df["age"] = df["age"].fillna(median_age).round().astype(int)

    # 6. Sales: missing -> price x quantity; extreme outliers -> price x quantity
    expected = df["price"] * df["quantity"]
    missing_sales = df["sales"].isna()
    df.loc[missing_sales, "sales"] = expected[missing_sales]
    report["missing_sales_imputed"] = int(missing_sales.sum())

    ratio = df["sales"] / expected
    outliers = (ratio > 3) | (ratio < 0.3)
    df.loc[outliers, "sales"] = expected[outliers]
    report["sales_outliers_corrected"] = int(outliers.sum())
    df["sales"] = df["sales"].round(2)

    # 7. Save
    df = df.sort_values("order_date").reset_index(drop=True)
    report["rows_clean"] = int(len(df))

    df.assign(order_date=df["order_date"].dt.strftime("%Y-%m-%d")).to_csv(DATA / "cleaned_orders.csv", index=False)
    (DATA / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Data cleaning complete")
    for key, value in report.items():
        print(f"  {key:<32} {value}")


if __name__ == "__main__":
    main()
