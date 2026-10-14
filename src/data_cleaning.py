

from pathlib import Path
import json
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load both sheets
print("Loading dataset...")
sheets = pd.read_excel(
    RAW_FILE,
    sheet_name=["Year 2009-2010", "Year 2010-2011"]
)

frames = []
for sheet_name, frame in sheets.items():
    frame = frame.copy()
    frame["SourceSheet"] = sheet_name
    frames.append(frame)

df = pd.concat(frames, ignore_index=True)
initial_rows = len(df)

# 2. Standardize column names
df.columns = [
    str(col).strip().replace(" ", "_")
    for col in df.columns
]

# 3. Standardize data types
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)
df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)
df["Customer_ID"] = pd.to_numeric(
    df["Customer_ID"], errors="coerce"
)

# 4. Define transaction columns for duplicate comparison.
# SourceSheet is metadata and must not distinguish duplicates.
duplicate_check_columns = [
    col for col in df.columns if col != "SourceSheet"
]

# Record quality issues BEFORE cleaning
report = {
    "initial_rows": int(initial_rows),
    "missing_customer_id": int(df["Customer_ID"].isna().sum()),
    "missing_description": int(df["Description"].isna().sum()),
    "invalid_invoice_date": int(df["InvoiceDate"].isna().sum()),
    "non_positive_quantity": int((df["Quantity"] <= 0).sum()),
    "non_positive_price": int((df["Price"] <= 0).sum()),
    "cancelled_invoice_rows": int(
        df["Invoice"].astype("string").str.startswith(
            "C", na=False
        ).sum()
    ),
    "exact_duplicate_rows": int(
        df.duplicated(subset=duplicate_check_columns).sum()
    ),
}

# 5. Remove exact duplicates using transaction columns only
df = df.drop_duplicates(
    subset=duplicate_check_columns
).copy()

# 6. Flag returns, cancellations and invalid prices
df["IsCancelledInvoice"] = (
    df["Invoice"].astype("string").str.startswith("C", na=False)
)
df["IsReturnOrNonPositiveQuantity"] = df["Quantity"] <= 0
df["IsNonPositivePrice"] = df["Price"] <= 0

# Transaction line value; this is NOT profit
df["LineValue"] = df["Quantity"] * df["Price"]

# 7. Create a positive-sales dataset
valid_sales = (
    ~df["IsCancelledInvoice"]
    & ~df["IsReturnOrNonPositiveQuantity"]
    & ~df["IsNonPositivePrice"]
    & df["InvoiceDate"].notna()
    & df["StockCode"].notna()
)

sales = df.loc[valid_sales].copy()

# 8. Complete the cleaning report
report["rows_after_duplicate_removal"] = int(len(df))
report["valid_sales_rows"] = int(len(sales))
report["rows_excluded_from_sales_dataset"] = int(
    len(df) - len(sales)
)
report["sales_revenue_before_customer_filter"] = round(
    float(sales["LineValue"].sum()), 2
)

# 9. Save processed files
df.to_csv(
    OUTPUT_DIR / "retail_transactions_cleaned.csv",
    index=False
)
sales.to_csv(
    OUTPUT_DIR / "retail_sales.csv",
    index=False
)

with open(
    OUTPUT_DIR / "cleaning_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=4)

# 10. Display results
print("\n--- DATA CLEANING REPORT ---")
for key, value in report.items():
    if isinstance(value, float):
        print(f"{key}: £{value:,.2f}")
    elif isinstance(value, int):
        print(f"{key}: {value:,}")
    else:
        print(f"{key}: {value}")

print("\n--- OUTPUT FILES ---")
print("retail_transactions_cleaned.csv")
print("retail_sales.csv")
print("cleaning_report.json")
print(f"Saved in: {OUTPUT_DIR}")
