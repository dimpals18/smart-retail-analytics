"""
Smart Retail Analytics Dashboard
--------------------------------
Run with:  streamlit run dashboard.py

Expects the outputs of the data pipeline inside the data/ folder:
    cleaned_orders.csv   (required)
    monthly_sales.csv    (optional - forecasting tab)
    sales_forecast.csv   (optional - forecasting tab)
    reviews.txt          (optional - reviews tab)
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = Path("data")
ORDERS_PATH = DATA_DIR / "cleaned_orders.csv"
MONTHLY_PATH = DATA_DIR / "monthly_sales.csv"
FORECAST_PATH = DATA_DIR / "sales_forecast.csv"
REVIEWS_PATH = DATA_DIR / "reviews.txt"
RFM_PATH = DATA_DIR / "rfm_segments.csv"
PARETO_PATH = DATA_DIR / "product_pareto.csv"
METRICS_PATH = DATA_DIR / "model_metrics.csv"
CLEANING_PATH = DATA_DIR / "cleaning_report.json"

REQUIRED_COLUMNS = {"order_id", "customer_id", "order_date", "sales", "city", "category"}

PRIMARY = "#2E86DE"
SECONDARY = "#F39C12"
POSITIVE = "#2ECC71"
NEGATIVE = "#E74C3C"
NEUTRAL = "#95A5A6"

st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS uses translucent colours so it looks right in both light and dark themes.
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 16px 18px;
            border-radius: 12px;
        }
        [data-testid="stMetricValue"] {font-weight: 800;}
        .stTabs [data-baseweb="tab-list"] {gap: 6px;}
        .stTabs [data-baseweb="tab"] {padding: 8px 18px; border-radius: 8px 8px 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def inr(value: float) -> str:
    """Format a number in Indian style: ₹1.25 Cr / ₹4.50 L / ₹12,345."""
    if pd.isna(value):
        return "–"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:.2f} L"
    return f"{sign}₹{v:,.0f}"


def pct_change(current: float, previous: float):
    """Return a '+12.3%' style string, or None if it can't be computed."""
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return f"{(current - previous) / previous:+.1%}"


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        title=dict(x=0, xanchor="left", y=0.97, yanchor="top"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


@st.cache_data(show_spinner=False)
def load_orders(path: str, modified: float) -> pd.DataFrame:
    """Load cleaned orders. `modified` busts the cache when the file changes."""
    df = pd.read_csv(path)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date", "sales"]).copy()
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[17, 25, 35, 45, 55, 65],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65"],
        )
    return df


@st.cache_data(show_spinner=False)
def load_csv_with_dates(path: str, modified: float, date_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = date_col or df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"})
    return df.sort_values("date")


@st.cache_data(show_spinner=False)
def load_reviews(path: str, modified: float) -> pd.DataFrame:
    """Read reviews.txt and tag each review with a simple keyword sentiment."""
    positive = {"excellent", "good", "great", "satisfied", "recommend", "exceeded", "fast", "value"}
    negative = {"delayed", "average", "not", "expensive", "poor", "bad"}
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = line.split(":", 1)[-1].strip()
        words = {w.strip(".,!?").lower() for w in text.split()}
        score = len(words & positive) - len(words & negative)
        sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        rows.append({"review": text, "sentiment": sentiment})
    return pd.DataFrame(rows)


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# ----------------------------------------------------------------------
# Header + data check
# ----------------------------------------------------------------------
st.title("📊 Smart Retail Analytics")
st.caption("Sales Intelligence  •  Customer Insights  •  Forecasting")

if not ORDERS_PATH.exists():
    st.error("Cleaned data not found. Run the data preparation scripts first.")
    st.code(
        "python create_data.py\n"
        "python src/data_collection.py\n"
        "python src/data_cleaning.py\n"
        "python src/database.py\n"
        "python src/analysis.py\n"
        "python src/forecasting.py",
        language="bash",
    )
    st.stop()

df = load_orders(str(ORDERS_PATH), mtime(ORDERS_PATH))

missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    st.error(f"cleaned_orders.csv is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

all_cities = sorted(df["city"].dropna().unique())
all_categories = sorted(df["category"].dropna().unique())

selected_cities = st.sidebar.multiselect("City", all_cities, default=all_cities)
selected_categories = st.sidebar.multiselect("Category", all_categories, default=all_categories)

min_date, max_date = df["order_date"].min().date(), df["order_date"].max().date()
selected_dates = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.info("Select an end date in the sidebar to continue.")
    st.stop()
if not selected_cities or not selected_categories:
    st.warning("Select at least one city and one category.")
    st.stop()

start, end = (pd.Timestamp(d) for d in selected_dates)
end_exclusive = end + pd.Timedelta(days=1)

# Everything except the date filter, so we can also build the "previous period".
scoped = df[df["city"].isin(selected_cities) & df["category"].isin(selected_categories)]
current = scoped[(scoped["order_date"] >= start) & (scoped["order_date"] < end_exclusive)]

if current.empty:
    st.warning("No orders match the selected filters.")
    st.stop()

span = end_exclusive - start
previous = scoped[(scoped["order_date"] >= start - span) & (scoped["order_date"] < start)]

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(current):,}** of {len(df):,} orders")

# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------
total_sales = current["sales"].sum()
total_orders = current["order_id"].nunique()
total_customers = current["customer_id"].nunique()
aov = total_sales / total_orders

prev_orders = previous["order_id"].nunique()
prev_sales = previous["sales"].sum() if not previous.empty else None
prev_aov = prev_sales / prev_orders if prev_orders else None
prev_customers = previous["customer_id"].nunique() if not previous.empty else None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", inr(total_sales), pct_change(total_sales, prev_sales), help="Change vs the previous period of equal length")
k2.metric("Orders", f"{total_orders:,}", pct_change(total_orders, prev_orders or None))
k3.metric("Customers", f"{total_customers:,}", pct_change(total_customers, prev_customers))
k4.metric("Avg Order Value", inr(aov), pct_change(aov, prev_aov))

st.write("")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_overview, tab_products, tab_forecast, tab_reviews, tab_data = st.tabs(
    ["📈 Overview", "🛍️ Products & Customers", "🔮 Forecast", "💬 Reviews", "🗂️ Data"]
)

# ---------------- Overview ----------------
with tab_overview:
    monthly_view = (
        current.groupby("month", as_index=False)
        .agg(sales=("sales", "sum"), orders=("order_id", "nunique"))
        .sort_values("month")
    )
    monthly_view["moving_avg_3m"] = monthly_view["sales"].rolling(3).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_view["month"], y=monthly_view["sales"], name="Monthly sales",
        mode="lines+markers", line=dict(color=PRIMARY, width=3),
        fill="tozeroy", fillcolor="rgba(46,134,222,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=monthly_view["month"], y=monthly_view["moving_avg_3m"], name="3-month moving average",
        mode="lines", line=dict(color=SECONDARY, width=2, dash="dash"),
    ))
    fig.update_layout(title="Monthly Sales Trend", yaxis_title="Sales (₹)")
    st.plotly_chart(style_fig(fig), width="stretch")

    col_a, col_b = st.columns(2)

    cat_sales = current.groupby("category", as_index=False)["sales"].sum().sort_values("sales")
    fig_cat = px.bar(cat_sales, x="sales", y="category", orientation="h",
                     title="Sales by Category", color_discrete_sequence=[PRIMARY])
    fig_cat.update_layout(xaxis_title="Sales (₹)", yaxis_title=None)
    col_a.plotly_chart(style_fig(fig_cat, 360), width="stretch")

    city_sales = current.groupby("city", as_index=False)["sales"].sum().sort_values("sales")
    fig_city = px.bar(city_sales, x="sales", y="city", orientation="h",
                      title="Sales by City", color_discrete_sequence=[SECONDARY])
    fig_city.update_layout(xaxis_title="Sales (₹)", yaxis_title=None)
    col_b.plotly_chart(style_fig(fig_city, 360), width="stretch")

    # Auto-generated insights
    top_cat = cat_sales.iloc[-1]
    top_city = city_sales.iloc[-1]
    best_month = monthly_view.loc[monthly_view["sales"].idxmax()]
    with st.container(border=True):
        st.markdown("**💡 Key insights**")
        st.markdown(
            f"- **{top_cat['category']}** is the top category with {inr(top_cat['sales'])} "
            f"({top_cat['sales'] / total_sales:.0%} of sales).\n"
            f"- **{top_city['city']}** leads all cities with {inr(top_city['sales'])}.\n"
            f"- Best month: **{best_month['month']:%B %Y}** ({inr(best_month['sales'])})."
        )

# ---------------- Products & Customers ----------------
with tab_products:
    product_col = "product_name" if "product_name" in current.columns else "product_id"
    if product_col in current.columns:
        top_products = (
            current.assign(**{product_col: current[product_col].astype(str)})
            .groupby(product_col, as_index=False)["sales"].sum()
            .nlargest(10, "sales").sort_values("sales")
        )
        fig_prod = px.bar(top_products, x="sales", y=product_col, orientation="h",
                          title="Top 10 Products by Sales", color_discrete_sequence=[PRIMARY])
        fig_prod.update_layout(xaxis_title="Sales (₹)", yaxis_title=None)

    col_a, col_b = st.columns(2)
    if product_col in current.columns:
        col_a.plotly_chart(style_fig(fig_prod, 400), width="stretch")

    heat = current.pivot_table(index="city", columns="category", values="sales", aggfunc="sum", fill_value=0)
    fig_heat = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                         title="Sales Heatmap: City × Category", text_auto=".2s")
    fig_heat.update_layout(coloraxis_showscale=False)
    col_b.plotly_chart(style_fig(fig_heat, 400), width="stretch")

    col_c, col_d = st.columns(2)
    if "age_group" in current.columns:
        age_sales = (
            current.groupby("age_group", observed=True, as_index=False)["sales"].sum()
        )
        fig_age = px.bar(age_sales, x="age_group", y="sales", title="Sales by Customer Age Group",
                         color_discrete_sequence=[SECONDARY])
        fig_age.update_layout(xaxis_title="Age group", yaxis_title="Sales (₹)")
        col_c.plotly_chart(style_fig(fig_age, 360), width="stretch")

    top_customers = (
        current.groupby("customer_id")
        .agg(total_spent=("sales", "sum"), orders=("order_id", "nunique"))
        .nlargest(10, "total_spent").reset_index()
    )
    top_customers["total_spent"] = top_customers["total_spent"].round(0)
    col_d.markdown("**🏆 Top 10 customers**")
    col_d.dataframe(top_customers, hide_index=True, width="stretch")

    st.divider()
    col_e, col_f = st.columns(2)

    if RFM_PATH.exists():
        rfm = pd.read_csv(RFM_PATH)
        seg = rfm.groupby("segment").agg(customers=("customer_id", "count"), avg_spend=("monetary", "mean")).reset_index()
        fig_rfm = px.bar(seg.sort_values("customers"), x="customers", y="segment", orientation="h",
                         title="Customer Segments (RFM)", color_discrete_sequence=[PRIMARY],
                         hover_data={"avg_spend": ":,.0f"})
        fig_rfm.update_layout(xaxis_title="Customers", yaxis_title=None)
        col_e.plotly_chart(style_fig(fig_rfm, 360), width="stretch")
    else:
        col_e.info("Run `python src/analysis.py` to generate RFM segments.")

    if PARETO_PATH.exists():
        pareto = pd.read_csv(PARETO_PATH)
        fig_p = go.Figure()
        fig_p.add_bar(x=pareto["product_name"], y=pareto["sales"], name="Sales", marker_color=PRIMARY)
        fig_p.add_scatter(x=pareto["product_name"], y=pareto["cumulative_share"] * 100, name="Cumulative %",
                          yaxis="y2", mode="lines+markers", line=dict(color=SECONDARY, width=2.5))
        fig_p.update_layout(title="Pareto: Product Sales (80/20 rule)",
                            yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False))
        fig_p.update_xaxes(showticklabels=False)
        col_f.plotly_chart(style_fig(fig_p, 360), width="stretch")
    else:
        col_f.info("Run `python src/analysis.py` to generate the Pareto analysis.")

# ---------------- Forecast ----------------
with tab_forecast:
    st.caption("Forecasts are built from the full history and are not affected by the sidebar filters.")

    if MONTHLY_PATH.exists():
        monthly = load_csv_with_dates(str(MONTHLY_PATH), mtime(MONTHLY_PATH))
        fig_f = go.Figure()
        series = [
            ("sales", "Actual sales", PRIMARY, "solid"),
            ("moving_average", "Moving average", SECONDARY, "dash"),
            ("exponential_smoothing", "Exponential smoothing", POSITIVE, "dot"),
        ]
        for col, label, colour, dash in series:
            if col in monthly.columns:
                fig_f.add_trace(go.Scatter(x=monthly["date"], y=monthly[col], name=label,
                                           mode="lines", line=dict(color=colour, dash=dash, width=2.5)))

        if FORECAST_PATH.exists():
            forecast = load_csv_with_dates(str(FORECAST_PATH), mtime(FORECAST_PATH), "date")
            fig_f.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast_sales"], name=f"Forecast ({forecast['model'].iloc[0]})" if "model" in forecast.columns else "Forecast",
                                       mode="lines+markers", line=dict(color=NEGATIVE, width=3, dash="dash")))
        fig_f.update_layout(title="Actual vs Model vs Forecast", yaxis_title="Sales (₹)")
        st.plotly_chart(style_fig(fig_f, 420), width="stretch")
    else:
        st.info("Run `python src/forecasting.py` to generate monthly_sales.csv.")

    if FORECAST_PATH.exists():
        forecast = load_csv_with_dates(str(FORECAST_PATH), mtime(FORECAST_PATH), "date")
        next_total = forecast["forecast_sales"].sum()
        m1, m2 = st.columns(2)
        m1.metric(f"Forecast for next {len(forecast)} months", inr(next_total))
        if MONTHLY_PATH.exists() and "sales" in monthly.columns:
            last_total = monthly["sales"].tail(len(forecast)).sum()
            m2.metric(f"Last {len(forecast)} months (actual)", inr(last_total),
                      pct_change(next_total, last_total), help="Forecast vs most recent actuals")

        shown = forecast.assign(date=forecast["date"].dt.strftime("%b %Y"))
        st.dataframe(shown, hide_index=True, width="stretch")
        if METRICS_PATH.exists():
            st.markdown("**Model comparison** (tested on the last 6 months; lower is better)")
            st.dataframe(pd.read_csv(METRICS_PATH), hide_index=True, width="stretch")
        st.download_button("⬇️ Download forecast", forecast.to_csv(index=False).encode("utf-8"),
                           "sales_forecast.csv", "text/csv")
    else:
        st.info("Run `python src/forecasting.py` to generate the forecast.")

# ---------------- Reviews ----------------
with tab_reviews:
    if REVIEWS_PATH.exists():
        reviews = load_reviews(str(REVIEWS_PATH), mtime(REVIEWS_PATH))
        counts = reviews["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)

        col_a, col_b = st.columns([1, 1])
        fig_s = px.pie(values=counts.values, names=counts.index, hole=0.55, title="Review Sentiment",
                       color=counts.index,
                       color_discrete_map={"Positive": POSITIVE, "Neutral": NEUTRAL, "Negative": NEGATIVE})
        fig_s.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        col_a.plotly_chart(fig_s, width="stretch")

        top_reviews = reviews["review"].value_counts().head(6).rename_axis("Review").reset_index(name="Count")
        col_b.markdown("**Most common feedback**")
        col_b.dataframe(top_reviews, hide_index=True, width="stretch")
        st.caption("Sentiment is a simple keyword-based score, suitable for a demo dataset.")
    else:
        st.info("reviews.txt not found in the data folder.")

# ---------------- Data ----------------
with tab_data:
    if CLEANING_PATH.exists():
        with st.expander("🧹 Data cleaning report"):
            rep_data = json.loads(CLEANING_PATH.read_text(encoding="utf-8"))
            labels = {k: k.replace("_", " ").capitalize() for k in rep_data}
            st.table(pd.DataFrame({"Step": labels.values(), "Count": rep_data.values()}).set_index("Step"))
    st.subheader("Cleaned Order Data")
    display = current.drop(columns=["month"], errors="ignore").sort_values("order_date", ascending=False)
    st.dataframe(display, hide_index=True, width="stretch")
    st.download_button("⬇️ Download filtered orders (CSV)", display.to_csv(index=False).encode("utf-8"),
                       "filtered_orders.csv", "text/csv")

st.divider()
st.caption("Smart Retail Analytics • Built with Streamlit, Pandas & Plotly")