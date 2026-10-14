
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "rfm_customers.csv"

def main():
    print("Loading cleaned retail sales data...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        parse_dates=["InvoiceDate"],
    )

    # Customer-level analysis requires a valid Customer ID
    df = df.dropna(subset=["Customer_ID"]).copy()

    # Keep only valid positive sales transactions
    df = df[
        (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (~df["IsCancelledInvoice"].astype(bool))
    ].copy()

    df["Customer_ID"] = df["Customer_ID"].astype("int64")

    # Monetary value per transaction line
    df["LineValue"] = df["Quantity"] * df["Price"]

    # Use one day after the latest transaction as the snapshot date
    snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    # Calculate Recency, Frequency and Monetary value
    rfm = df.groupby("Customer_ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineValue", "sum"),
    )

    rfm["Recency"] = (
        snapshot_date - rfm["LastPurchase"].dt.normalize()
    ).dt.days

    # Assign quintile scores: 1 to 5
    # Lower recency is better; higher frequency/monetary is better
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    # Initial interpretable customer segments
    def segment_customer(row):
        r, f = row["R_Score"], row["F_Score"]

        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 4:
            return "Loyal Customers"
        elif r >= 4 and f >= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2:
            return "Lost Customers"
        else:
            return "Needs Attention"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)

    # Sort by monetary value, highest first
    rfm = rfm.sort_values("Monetary", ascending=False).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(OUTPUT_FILE, index=False)

    print("\nRFM analysis completed successfully.")
    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Customers analyzed: {len(rfm):,}")
    print(f"Total customer monetary value: £{rfm['Monetary'].sum():,.2f}")

    print("\nCustomer segments:")
    print(rfm["Segment"].value_counts().to_string())

    print(f"\nTop 10 customers by monetary value:")
    print(
        rfm[["Customer_ID", "Recency", "Frequency", "Monetary", "Segment"]]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
