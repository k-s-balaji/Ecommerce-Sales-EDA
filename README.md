# E-Commerce Sales Performance and Customer Analysis

An exploratory analysis of invoice-level sales from the UCI Online Retail dataset. The project combines transaction cleaning, monthly performance analysis, product ranking, seasonality, customer cohorts, and PostgreSQL queries.

## Dataset

The source contains 541,909 invoice-line records from a UK online retailer, dated 1 December 2010 through 9 December 2011. Dataset: [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail). Citation: Chen, D. (2015), *Online Retail*, DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33). License: CC BY 4.0.

## Method

Exact duplicate lines, cancellation invoices, invalid dates, and non-positive quantity or price lines are excluded from positive-sales outputs. IQR bounds are calculated for quantity, unit price, and line revenue; flagged high values are retained because wholesale orders may be valid. Sales lines without customer IDs remain in revenue totals but are excluded from customer and cohort measures.

## Results

The cleaned sales table contains 524,878 positive lines across 19,960 invoices, with £10,642,110.80 in positive line sales. Of 4,338 identified customers, 65.58% placed at least two invoices during the observation window. November 2011 is the highest complete month at £1.50 million. December ends on the 9th and is partial.

Customer ID is missing on 132,186 retained lines. Cohort retention is based on identified customers only, and later cohorts have fewer months available for comparison. Postage and service codes are excluded from the top merchandise chart but remain in sales totals.

## Files

- `analyze_sales.py` — Pandas and NumPy cleaning, aggregation, cohort analysis, and chart generation.
- `sql/sales_analysis.sql` — PostgreSQL table definitions and queries for monthly sales, customer repeat behavior, cohorts, products, and countries.
- `results/figures/` — monthly sales, weekday and month seasonality, cohort retention, and top merchandise items.
- `results/analysis_summary.json` — headline measures and IQR bounds.
- `data/processed/` — prepared analysis tables, generated locally from the source workbook.

The source workbook is retrieved from UCI when it is not present in `data/raw/`. Raw and processed data are excluded from version control.
