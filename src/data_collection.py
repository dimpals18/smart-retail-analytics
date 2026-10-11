"""
Step 1 - Data collection.

Reads the raw files (CSV, JSON, XML, TXT), cross-checks the two product
sources, joins everything into one raw table and saves it.
"""
from pathlib import Path

import pandas as pd

DATA = Path("data")


def main() -> None:
    customers = pd.read_csv(DATA / "customers.csv")
    orders = pd.read_csv(DATA / "orders.csv")
    products_json = pd.read_json(DATA / "products.json")
    products_xml = pd.read_xml(DATA / "products.xml")

    # Reviews (plain text): "Review 12: text..."
    lines = (DATA / "reviews.txt").read_text(encoding="utf-8").splitlines()
    reviews = pd.DataFrame(
        [{"review_id": i, "text": line.split(":", 1)[-1].strip()} for i, line in enumerate(lines, 1) if line.strip()]
    )
    reviews.to_csv(DATA / "reviews.csv", index=False)

    # Cross-check JSON vs XML product sources
    check = products_json.merge(products_xml, on="product_id", suffixes=("_json", "_xml"))
    mismatches = int((check["price_json"] != check["price_xml"]).sum())

    # Join into one raw table (left joins keep every order)
    raw = orders.merge(customers, on="customer_id", how="left").merge(products_json, on="product_id", how="left")
    raw.to_csv(DATA / "raw_orders.csv", index=False)

    print("Data collection complete")
    print(f"  customers.csv   : {len(customers):>5} rows")
    print(f"  products.json   : {len(products_json):>5} rows")
    print(f"  products.xml    : {len(products_xml):>5} rows ({mismatches} price mismatches vs JSON)")
    print(f"  orders.csv      : {len(orders):>5} rows")
    print(f"  reviews.txt     : {len(reviews):>5} rows")
    print(f"  -> data/raw_orders.csv ({len(raw)} rows, {raw.shape[1]} columns)")


if __name__ == "__main__":
    main()
