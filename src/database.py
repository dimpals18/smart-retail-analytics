"""
Step 3 - Load the cleaned data into a SQLite database (normalised tables).
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DATA = Path("data")
DB_URL = f"sqlite:///{(DATA / 'retail.db').as_posix()}"


def main() -> None:
    df = pd.read_csv(DATA / "cleaned_orders.csv")

    customers = df[["customer_id", "customer_name", "age", "city"]].drop_duplicates("customer_id")
    products = df[["product_id", "product_name", "category", "price"]].drop_duplicates("product_id")
    orders = df[["order_id", "customer_id", "product_id", "order_date", "quantity", "sales"]]

    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)"))

    with engine.connect() as conn:
        print("Database created: data/retail.db")
        for table in ("customers", "products", "orders"):
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table:<10} {count:>5} rows")


if __name__ == "__main__":
    main()
