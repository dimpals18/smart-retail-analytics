"""
Step 4 - Analysis using SQL queries on the database.

Outputs (in data/):
    category_summary.csv  - sales by category
    product_pareto.csv    - 80/20 (ABC) product analysis
    rfm_segments.csv      - RFM customer segmentation
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

DATA = Path("data")
engine = create_engine(f"sqlite:///{(DATA / 'retail.db').as_posix()}")


def query(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, engine)


def category_summary() -> pd.DataFrame:
    return query("""
        SELECT p.category,
               COUNT(DISTINCT o.order_id) AS orders,
               SUM(o.quantity)            AS units,
               ROUND(SUM(o.sales), 2)     AS sales
        FROM orders o JOIN products p ON o.product_id = p.product_id
        GROUP BY p.category ORDER BY sales DESC
    """)


def product_pareto() -> pd.DataFrame:
    df = query("""
        SELECT p.product_id, p.product_name, p.category, ROUND(SUM(o.sales), 2) AS sales
        FROM orders o JOIN products p ON o.product_id = p.product_id
        GROUP BY p.product_id ORDER BY sales DESC
    """)
    df["cumulative_share"] = (df["sales"].cumsum() / df["sales"].sum()).round(4)
    df["abc_class"] = pd.cut(df["cumulative_share"], [0, 0.8, 0.95, 1.01], labels=["A", "B", "C"])
    return df


def rfm_segments() -> pd.DataFrame:
    df = query("""
        SELECT customer_id,
               MAX(order_date)           AS last_order,
               COUNT(DISTINCT order_id)  AS frequency,
               ROUND(SUM(sales), 2)      AS monetary
        FROM orders GROUP BY customer_id
    """)
    df["last_order"] = pd.to_datetime(df["last_order"])
    snapshot = df["last_order"].max() + pd.Timedelta(days=1)
    df["recency_days"] = (snapshot - df["last_order"]).dt.days

    # 1-5 scores (5 = best). rank(method="first") avoids duplicate bin edges.
    df["R"] = pd.qcut(df["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M"] = pd.qcut(df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def label(row) -> str:
        if row.R >= 4 and row.F >= 4:
            return "Champions"
        if row.F >= 4:
            return "Loyal"
        if row.R <= 2 and row.F >= 3:
            return "At Risk"
        if row.R <= 2:
            return "Lost"
        return "Promising"

    df["segment"] = df.apply(label, axis=1)
    return df.drop(columns="last_order")


def main() -> None:
    cat = category_summary()
    pareto = product_pareto()
    rfm = rfm_segments()

    cat.to_csv(DATA / "category_summary.csv", index=False)
    pareto.to_csv(DATA / "product_pareto.csv", index=False)
    rfm.to_csv(DATA / "rfm_segments.csv", index=False)

    print("Analysis complete")
    print(f"  Top category : {cat.iloc[0]['category']} (Rs {cat.iloc[0]['sales']:,.0f})")
    n_a = int((pareto['abc_class'] == 'A').sum())
    print(f"  Pareto       : {n_a} of {len(pareto)} products make up ~80% of sales")
    print("  RFM segments :", rfm["segment"].value_counts().to_dict())


if __name__ == "__main__":
    main()
