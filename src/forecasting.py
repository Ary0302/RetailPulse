
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

def prepare_daily_sales():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Sales data not found: {INPUT_FILE}. "
            "Run src/data_cleaning.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the cleaned sales data
    sales = pd.read_csv(
        INPUT_FILE,
        parse_dates=["InvoiceDate"]
    )

    # Keep only valid dates and calculate sales value
    sales = sales.dropna(subset=["InvoiceDate", "Quantity", "Price"])
    sales["LineValue"] = sales["Quantity"] * sales["Price"]

    # Aggregate revenue and quantity by calendar day
    daily = (
        sales.set_index("InvoiceDate")
        .sort_index()
        .resample("D")
        .agg(
            Revenue=("LineValue", "sum"),
            Demand=("Quantity", "sum"),
            Orders=("Invoice", "nunique")
        )
    )

    # Add missing calendar dates with zero activity
    daily = daily.asfreq("D", fill_value=0)
    daily.index.name = "Date"

    # Save the time series for model development
    output_file = OUTPUT_DIR / "daily_sales.csv"
    daily.to_csv(output_file)

    print("Daily sales dataset created successfully.")
    print(f"Output: {output_file}")
    print(f"Date range: {daily.index.min()} to {daily.index.max()}")
    print(f"Number of days: {len(daily)}")
    print("\nFirst five rows:")
    print(daily.head())
    print("\nSummary statistics:")
    print(daily[["Revenue", "Demand", "Orders"]].describe())

if __name__ == "__main__":
    prepare_daily_sales()
