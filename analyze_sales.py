from pathlib import Path
import json
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"
raw_folder = data_folder / "raw"
processed_folder = data_folder / "processed"
results_folder = project_folder / "results"
figures_folder = results_folder / "figures"
source_file = raw_folder / "Online Retail.xlsx"

for folder in [raw_folder, processed_folder, results_folder, figures_folder]:
    folder.mkdir(parents=True, exist_ok=True)

# Download the public workbook if it is not already in the data folder.
if not source_file.exists():
    zip_file = raw_folder / "online_retail.zip"
    source_url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    urllib.request.urlretrieve(source_url, zip_file)

    with zipfile.ZipFile(zip_file, "r") as retail_zip:
        for file_name in retail_zip.namelist():
            if file_name.endswith("Online Retail.xlsx"):
                source_file.write_bytes(retail_zip.read(file_name))
                break

    zip_file.unlink()

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.titlecolor"] = "#16324F"
plt.rcParams["text.color"] = "#16324F"

# Read the workbook and make the column names easier to work with.
sales_data = pd.read_excel(source_file)
sales_data.columns = sales_data.columns.str.lower().str.replace(" ", "_")
sales_data["invoicedate"] = pd.to_datetime(sales_data["invoicedate"], errors="coerce")
sales_data["customerid"] = sales_data["customerid"].astype("Int64")
sales_data["revenue"] = sales_data["quantity"] * sales_data["unitprice"]

source_row_count = len(sales_data)
duplicate_count = int(sales_data.duplicated().sum())
cancelled_count = int(sales_data["invoiceno"].astype(str).str.startswith("C").sum())
missing_customer_count = int(sales_data["customerid"].isna().sum())

# Keep positive sales lines. Cancelled invoices, invalid dates, and returns are left out.
cancelled_invoices = sales_data["invoiceno"].astype(str).str.startswith("C")
bad_dates = sales_data["invoicedate"].isna()
non_positive_quantity = sales_data["quantity"] <= 0
non_positive_price = sales_data["unitprice"] <= 0

sales = sales_data.loc[
    ~(cancelled_invoices | bad_dates | non_positive_quantity | non_positive_price)
].drop_duplicates().copy()

sales["invoiceno"] = sales["invoiceno"].astype(str).str.strip()
sales["invoice_date"] = sales["invoicedate"].dt.date
sales["month"] = sales["invoicedate"].dt.to_period("M").dt.to_timestamp()
sales["description"] = sales["description"].fillna("(Unknown)").astype(str).str.strip()
sales["stockcode"] = sales["stockcode"].astype(str).str.strip()
sales["country"] = sales["country"].astype(str).str.strip()

# Count IQR outliers for quantity, price, and line revenue.
iqr_results = {}
for column in ["quantity", "unitprice", "revenue"]:
    lower_quartile = sales[column].quantile(0.25)
    upper_quartile = sales[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    lower_bound = lower_quartile - 1.5 * iqr
    upper_bound = upper_quartile + 1.5 * iqr
    outlier_count = int(((sales[column] < lower_bound) | (sales[column] > upper_bound)).sum())

    iqr_results[column] = {
        "q1": float(lower_quartile),
        "q3": float(upper_quartile),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_rows": outlier_count,
    }

# Monthly totals and average order value.
monthly_sales = sales.groupby("month").agg(
    revenue=("revenue", "sum"),
    orders=("invoiceno", "nunique"),
    customers=("customerid", "nunique"),
    units=("quantity", "sum"),
).reset_index()
monthly_sales["average_order_value"] = monthly_sales["revenue"] / monthly_sales["orders"].replace(0, np.nan)
monthly_sales.to_csv(processed_folder / "monthly_sales.csv", index=False)

# Customer totals and simple repeat-purchase groups.
customer_sales = sales.dropna(subset=["customerid"])
customer_summary = customer_sales.groupby("customerid").agg(
    first_order=("invoicedate", "min"),
    last_order=("invoicedate", "max"),
    orders=("invoiceno", "nunique"),
    revenue=("revenue", "sum"),
    units=("quantity", "sum"),
    country=("country", "last"),
).reset_index()

revenue_cutoff = customer_summary["revenue"].quantile(0.75)
customer_summary["segment"] = "One-time"
customer_summary.loc[customer_summary["orders"] >= 2, "segment"] = "Repeat"
customer_summary.loc[
    (customer_summary["orders"] >= 5) & (customer_summary["revenue"] >= revenue_cutoff),
    "segment",
] = "High-value repeat"
customer_summary["recency_days"] = (
    sales["invoicedate"].max() - customer_summary["last_order"]
).dt.days

# One row per customer invoice, used for order-level reporting.
customer_orders = sales.dropna(subset=["customerid"]).groupby(
    ["customerid", "invoiceno", "month"]
).agg(
    order_revenue=("revenue", "sum"),
    order_units=("quantity", "sum"),
    country=("country", "last"),
).reset_index()
customer_orders.to_csv(processed_folder / "fact_orders.csv", index=False)

# Monthly cohort retention for customers with an ID.
sales["cohort_month"] = sales.groupby("customerid")["month"].transform("min")
sales["cohort_number"] = (
    (sales["month"].dt.year - sales["cohort_month"].dt.year) * 12
    + sales["month"].dt.month
    - sales["cohort_month"].dt.month
    + 1
)

cohort_data = sales.dropna(subset=["customerid"]).groupby(
    ["cohort_month", "cohort_number"]
)["customerid"].nunique().reset_index(name="customers")
cohort_sizes = cohort_data.loc[
    cohort_data["cohort_number"] == 1, ["cohort_month", "customers"]
].rename(columns={"customers": "cohort_size"})
cohort_data = cohort_data.merge(cohort_sizes, on="cohort_month")
cohort_data["retention"] = cohort_data["customers"] / cohort_data["cohort_size"]
cohort_data.to_csv(processed_folder / "cohort_retention.csv", index=False)
cohort_table = cohort_data.pivot(
    index="cohort_month", columns="cohort_number", values="retention"
)

# Product and country totals.
product_sales = sales.groupby(["stockcode", "description"]).agg(
    revenue=("revenue", "sum"),
    units=("quantity", "sum"),
    orders=("invoiceno", "nunique"),
).reset_index()

country_sales = sales.groupby("country").agg(
    revenue=("revenue", "sum"),
    orders=("invoiceno", "nunique"),
    customers=("customerid", "nunique"),
).reset_index()
country_sales.to_csv(processed_folder / "country_sales.csv", index=False)

# Calendar table for the Power BI model.
date_table = pd.DataFrame({
    "date": pd.date_range(sales["invoicedate"].min().date(), sales["invoicedate"].max().date())
})
date_table["year"] = date_table["date"].dt.year
date_table["quarter"] = "Q" + date_table["date"].dt.quarter.astype(str)
date_table["month"] = date_table["date"].dt.month
date_table["month_name"] = date_table["date"].dt.strftime("%B")
date_table["month_start"] = date_table["date"].dt.to_period("M").dt.to_timestamp()
date_table["weekday"] = date_table["date"].dt.day_name()
date_table["weekday_num"] = date_table["date"].dt.dayofweek + 1
date_table.to_csv(processed_folder / "dim_date.csv", index=False)

product_dimension = product_sales[["stockcode", "description"]].drop_duplicates()
product_dimension.to_csv(processed_folder / "dim_product.csv", index=False)
country_dimension = country_sales[["country"]].drop_duplicates()
country_dimension.to_csv(processed_folder / "dim_country.csv", index=False)
customer_dimension = customer_summary[["customerid", "country", "first_order", "orders", "segment"]]
customer_dimension.to_csv(processed_folder / "dim_customer.csv", index=False)

sales_fact = sales[[
    "invoiceno", "stockcode", "customerid", "invoicedate", "invoice_date",
    "quantity", "unitprice", "revenue", "country",
]]
sales_fact.to_csv(processed_folder / "fact_sales.csv", index=False)

# Monthly sales chart.
fig, ax = plt.subplots(figsize=(12, 5.8))
ax.plot(monthly_sales["month"], monthly_sales["revenue"], color="#176B87", marker="o", linewidth=2.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000_000:.1f}m"))
ax.set_ylabel("Positive line sales")
ax.set_title("Monthly sales", loc="left", fontsize=16, fontweight="bold")
ax.set_xlabel("")
ax.grid(axis="x", visible=False)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="both", length=0, pad=8)
ax.set_xticks(monthly_sales["month"])
ax.set_xticklabels(monthly_sales["month"].dt.strftime("%b\n%Y"))
peak_month = monthly_sales.loc[monthly_sales["revenue"].idxmax()]
ax.annotate(
    f"Peak £{peak_month['revenue'] / 1_000_000:.2f}m",
    (peak_month["month"], peak_month["revenue"]),
    xytext=(-8, 12), textcoords="offset points", ha="right", fontweight="bold",
)
last_date = sales["invoicedate"].max()
if last_date.day < last_date.days_in_month:
    last_sales = monthly_sales.iloc[-1]
    ax.scatter(last_sales["month"], last_sales["revenue"], color="#E49B28", s=65, zorder=3)
    ax.annotate("Partial month", (last_sales["month"], last_sales["revenue"]),
                xytext=(-4, 12), textcoords="offset points", ha="right", color="#8A5A12")
fig.text(0.08, 0.92, "Invoice line revenue · GBP", fontsize=9, color="#667085")
fig.tight_layout()
fig.savefig(figures_folder / "monthly_revenue.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Sales by weekday and month.
sales["month_name_short"] = sales["invoicedate"].dt.strftime("%b")
sales["weekday"] = sales["invoicedate"].dt.day_name()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
season_table = sales.pivot_table(
    index="weekday", columns="month_name_short", values="revenue", aggfunc="sum", fill_value=0
)
season_table = season_table.reindex(index=weekday_order, columns=month_order, fill_value=0)
fig, ax = plt.subplots(figsize=(12, 5.5))
sns.heatmap(season_table / 1000, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.7, ax=ax)
ax.set_title("Sales seasonality", loc="left", fontsize=16, fontweight="bold")
ax.set_xlabel("Invoice month")
ax.set_ylabel("")
fig.text(0.08, 0.92, "Monthly line revenue by weekday · £000s · no Saturday invoices in the source",
         fontsize=9, color="#667085")
fig.tight_layout()
fig.savefig(figures_folder / "seasonality_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Retention by month of first purchase.
cohort_sizes = cohort_data.loc[cohort_data["cohort_number"] == 1].set_index("cohort_month")["cohort_size"]
cohort_labels = [
    f"{month:%b %Y} (n={int(cohort_sizes.loc[month]):,})" for month in cohort_table.index
]
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    cohort_table, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1,
    yticklabels=cohort_labels, linewidths=0.7, ax=ax,
)
ax.set_title("Customer cohort retention", loc="left", fontsize=16, fontweight="bold")
ax.set_xlabel("Months since first purchase")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(figures_folder / "cohort_retention.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Top merchandise items. Postage and service lines are left out of this chart.
service_items = [
    "DOTCOM POSTAGE", "POSTAGE", "MANUAL", "BANK CHARGES",
    "AMAZON FEE", "CRUK COMMISSION", "ADJUST BAD DEBT",
]
merchandise_sales = product_sales.loc[
    ~product_sales["description"].str.upper().isin(service_items)
]
top_products = merchandise_sales.nlargest(10, "revenue").sort_values("revenue")
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.barh(top_products["description"].str.slice(0, 44), top_products["revenue"], color="#176B87")
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1000:.0f}k"))
ax.set_title("Top 10 merchandise items", loc="left", fontsize=16, fontweight="bold")
ax.set_xlabel("Positive line sales")
ax.set_ylabel("")
ax.grid(axis="y", visible=False)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(figures_folder / "top_products.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Save the main cleaning and sales results.
results = {
    "source_rows": source_row_count,
    "exact_duplicate_rows": duplicate_count,
    "cancellation_invoice_rows": cancelled_count,
    "missing_customer_ids": missing_customer_count,
    "clean_sales_rows": len(sales),
    "clean_orders": int(sales["invoiceno"].nunique()),
    "identified_customers": int(sales["customerid"].nunique()),
    "unassigned_customer_rows_retained_in_sales": int(sales["customerid"].isna().sum()),
    "date_min": str(sales["invoicedate"].min().date()),
    "date_max": str(sales["invoicedate"].max().date()),
    "positive_sales_line_revenue_gbp": round(float(sales["revenue"].sum()), 2),
    "repeat_customer_share": round(float((customer_summary["orders"] >= 2).mean()), 4),
    "one_time_customer_share": round(float((customer_summary["orders"] == 1).mean()), 4),
    "high_value_repeat_customers": int((customer_summary["segment"] == "High-value repeat").sum()),
    "iqr_diagnostics": iqr_results,
}

with open(results_folder / "analysis_summary.json", "w", encoding="utf-8") as summary_file:
    json.dump(results, summary_file, indent=2)

with open(results_folder / "cleaning_log.csv", "w", encoding="utf-8") as log_file:
    log_file.write("step,rows\n")
    log_file.write(f"source_rows,{source_row_count}\n")
    log_file.write(f"exact_duplicate_rows,{duplicate_count}\n")
    log_file.write(f"cancellation_invoice_rows,{cancelled_count}\n")
    log_file.write(f"clean_sales_rows,{len(sales)}\n")
    log_file.write(f"missing_customer_id_rows_retained_in_sales,{sales['customerid'].isna().sum()}\n")

print(json.dumps(results, indent=2))
print("\nMonthly sales:")
print(monthly_sales.to_string(index=False))
