"""
Step 0 - Generate a synthetic retail dataset.

The data is deliberately a little messy (duplicates, inconsistent city names,
missing / impossible values, outliers) so that the cleaning step in
src/data_cleaning.py has real problems to solve.
"""
import os

import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
np.random.seed(42)
rng = np.random.default_rng(42)  # separate generator used only for the "dirty" injections

# ---------------- CUSTOMERS ----------------
num_customers = 200
cities = [
    "Nagpur", "Mumbai", "Pune", "Delhi", "Bengaluru",
    "Hyderabad", "Chennai", "Jaipur", "Indore", "Nashik",
]

customers = pd.DataFrame({
    "customer_id": range(1, num_customers + 1),
    "customer_name": [f"Customer_{i}" for i in range(1, num_customers + 1)],
    "age": np.random.randint(18, 66, num_customers).astype(float),
    "city": np.random.choice(cities, num_customers),
})

# Dirty data: missing ages and impossible ages
customers.loc[customers.sample(5, random_state=42).index, "age"] = np.nan
customers.loc[customers.sample(3, random_state=7).index, "age"] = [120, 0, 150]

# Dirty data: inconsistent city spelling (case / extra spaces)
messy_idx = customers.sample(15, random_state=11).index
for n, idx in enumerate(messy_idx):
    city = customers.at[idx, "city"]
    customers.at[idx, "city"] = [city.lower(), city.upper(), f" {city} "][n % 3]

customers.to_csv("data/customers.csv", index=False)

# ---------------- PRODUCTS ----------------
num_products = 30
categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Accessories"]

products = pd.DataFrame({
    "product_id": range(101, 101 + num_products),
    "product_name": [f"Product_{i}" for i in range(1, num_products + 1)],
    "category": np.random.choice(categories, num_products),
    "price": np.random.randint(150, 10001, num_products),
})

products.to_json("data/products.json", orient="records", indent=2)
products.to_xml("data/products.xml", index=False, root_name="products")

# ---------------- ORDERS ----------------
num_orders = 3600
start_date = pd.Timestamp("2023-10-01")
end_date = pd.Timestamp("2026-10-01")

order_dates = pd.to_datetime(
    np.random.randint(start_date.value // 10**9, end_date.value // 10**9, num_orders),
    unit="s",
).normalize()

orders = pd.DataFrame({
    "order_id": range(10001, 10001 + num_orders),
    "customer_id": np.random.randint(1, num_customers + 1, num_orders),
    "product_id": np.random.randint(101, 101 + num_products, num_orders),
    "order_date": order_dates,
    "quantity": np.random.randint(1, 6, num_orders),
})
# On Windows numpy integers default to 32-bit; use 64-bit so later edits never fail
for col in ("order_id", "customer_id", "product_id", "quantity"):
    orders[col] = orders[col].astype("int64")

price_lookup = products.set_index("product_id")["price"]
orders["base_sales"] = orders["product_id"].map(price_lookup) * orders["quantity"]

# Monthly growth trend + seasonal variation + random noise
month_number = (orders["order_date"].dt.year - 2023) * 12 + orders["order_date"].dt.month - 10
seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * orders["order_date"].dt.month / 12)
growth_factor = 1 + 0.008 * month_number
random_factor = np.random.uniform(0.85, 1.15, num_orders)

orders["sales"] = (orders["base_sales"] * seasonal_factor * growth_factor * random_factor).round(2)
orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")
orders = orders.drop(columns=["base_sales"])

# Dirty data: missing sales, outliers (data-entry errors), invalid quantities
orders.loc[orders.sample(15, random_state=1).index, "sales"] = np.nan
outlier_idx = orders.sample(8, random_state=2).index
orders.loc[outlier_idx, "sales"] = orders.loc[outlier_idx, "sales"] * 10
orders.loc[orders.sample(6, random_state=3).index, "quantity"] = np.array([0, -1, 0, -2, 0, -1], dtype="int64")

# Dirty data: duplicate order rows
orders = pd.concat([orders, orders.sample(40, random_state=4)], ignore_index=True)
orders = orders.sample(frac=1, random_state=5).reset_index(drop=True)

orders.to_csv("data/orders.csv", index=False)

# ---------------- REVIEWS ----------------
review_texts = [
    "Excellent product and fast delivery",
    "Good quality for the price",
    "Average product, could be better",
    "Very satisfied with the purchase",
    "Delivery was delayed",
    "Great value for money",
    "Product quality exceeded expectations",
    "Not satisfied with the packaging",
    "Would recommend this product",
    "Good product but expensive",
]
with open("data/reviews.txt", "w", encoding="utf-8") as file:
    for i in range(1, 301):
        file.write(f"Review {i}: {np.random.choice(review_texts)}\n")

print("Retail dataset generated successfully!")
print(f"Customers: {len(customers)} | Products: {len(products)} | Orders: {len(orders)} (incl. duplicates) | Reviews: 300")
print("Historical period: October 2023 to September 2026")
