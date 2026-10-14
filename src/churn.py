
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "churn_training_data.csv"

LOOKBACK_DAYS = 365
CHURN_WINDOW_DAYS = 90


def main():
    print("Loading retail sales data...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        parse_dates=["InvoiceDate"],
    )

    # Customer-level analysis requires a known customer ID.
    df = df.dropna(subset=["Customer_ID"]).copy()
    df["Customer_ID"] = df["Customer_ID"].astype("int64")
    df["LineValue"] = df["Quantity"] * df["Price"]

    # Last available calendar day, expressed as an exclusive boundary.
    data_end = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    # Reserve the final 90 days to determine churn labels.
    snapshot_date = data_end - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    history_start = snapshot_date - pd.Timedelta(days=LOOKBACK_DAYS)

    # Historical purchases are used only for model features.
    history = df[
        (df["InvoiceDate"] >= history_start)
        & (df["InvoiceDate"] < snapshot_date)
    ].copy()

    # Future purchases are used only to create the label.
    future = df[
        (df["InvoiceDate"] >= snapshot_date)
        & (df["InvoiceDate"] < data_end)
    ].copy()

    # Customers must have purchased in the historical window.
    features = history.groupby("Customer_ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineValue", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        TotalQuantity=("Quantity", "sum"),
    )

    features["Recency"] = (
        snapshot_date - features["LastPurchase"]
    ).dt.days

    features["AvgOrderValue"] = (
        features["Monetary"] / features["Frequency"]
    )

    # Any purchase in the future window means not churned.
    future_customers = set(future["Customer_ID"].unique())
    features["Churn"] = (
        ~features.index.isin(future_customers)
    ).astype(int)

    features = features.reset_index()

    # Basic checks before saving.
    assert features["Customer_ID"].is_unique
    assert features["Churn"].isin([0, 1]).all()
    assert features.notna().all().all()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_FILE, index=False)

    print("\nChurn dataset created successfully.")
    print(f"Historical window: {history_start.date()} to {(snapshot_date - pd.Timedelta(days=1)).date()}")
    print(f"Future window: {snapshot_date.date()} to {(data_end - pd.Timedelta(days=1)).date()}")
    print(f"Customers: {len(features):,}")
    print("\nChurn label counts (0 = active, 1 = churned):")
    print(features["Churn"].value_counts().sort_index().to_string())
    print("\nChurn label percentages:")
    print(
        (features["Churn"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
        .to_string()
    )
    print(f"\nSaved dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

