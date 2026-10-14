
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
SALES_FILE = DATA_DIR / "retail_sales.csv"

st.set_page_config(
    page_title="RetailPulse | Customer & Sales Analytics",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# 2. Data loading
# --------------------------------------------------
@st.cache_data
def load_sales(file_path, modified_time):
    sales = pd.read_csv(
        file_path,
        parse_dates=["InvoiceDate"],
        dtype={"Invoice": "string", "StockCode": "string"},
        low_memory=False,
    )
    sales["LineValue"] = sales["Quantity"] * sales["Price"]
    return sales


# --------------------------------------------------
# 3. Sidebar
# --------------------------------------------------
st.sidebar.title("🛍️ RetailPulse")
st.sidebar.caption("AI-Powered Retail Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Sales Overview",
        "RFM Analytics",
        "Customer Segmentation",
        "Churn Prediction",
        "Demand Forecasting",
        "Inventory Insights",
    ],
)

st.sidebar.divider()
st.sidebar.caption("RetailPulse MVP | Historical retail data")

st.title("RetailPulse Dashboard")
st.write(
    "Explore sales performance, customer behaviour, "
    "churn risk and historical demand."
)

if not SALES_FILE.exists():
    st.error(f"Sales dataset not found: {SALES_FILE}")
    st.info("Run python src\\data_cleaning.py from the project root.")
    st.stop()

sales = load_sales(str(SALES_FILE), SALES_FILE.stat().st_mtime)


# --------------------------------------------------
# 4. Sales Overview
# --------------------------------------------------
if page == "Sales Overview":
    total_revenue = sales["LineValue"].sum()
    total_units = sales["Quantity"].sum()
    total_orders = sales["Invoice"].nunique()
    total_products = sales["StockCode"].nunique()
    identified_customers = sales["Customer_ID"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
    col2.metric("Units Sold", f"{total_units:,.0f}")
    col3.metric("Total Orders", f"{total_orders:,}")

    col4, col5 = st.columns(2)
    col4.metric("Unique Products", f"{total_products:,}")
    col5.metric("Identified Customers", f"{identified_customers:,}")

    st.subheader("Monthly Revenue")
    monthly_revenue = (
        sales.set_index("InvoiceDate")["LineValue"]
        .resample("MS")
        .sum()
    )
    monthly_revenue.index = monthly_revenue.index.strftime("%Y-%m")
    st.line_chart(monthly_revenue)

    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 Products by Units Sold")
        top_products = (
            sales.groupby("StockCode")
            .agg(
                UnitsSold=("Quantity", "sum"),
                Description=("Description", "first"),
            )
            .nlargest(10, "UnitsSold")
            .sort_values("UnitsSold")
        )
        top_products["Product"] = (
            top_products["Description"]
            .fillna(top_products.index.to_series())
            .astype(str)
            .str.slice(0, 45)
        )
        st.bar_chart(top_products.set_index("Product")["UnitsSold"])

    with right:
        st.subheader("Top 10 Countries by Revenue")
        country_revenue = (
            sales.groupby("Country")["LineValue"]
            .sum()
            .nlargest(10)
            .sort_values()
        )
        st.bar_chart(country_revenue)

    st.caption(
        "Revenue is sales value, not profit. Customer counts exclude "
        "transactions with missing customer IDs."
    )


# --------------------------------------------------
# 5. RFM Analytics
# --------------------------------------------------
elif page == "RFM Analytics":
    st.header("RFM Customer Analytics")
    rfm_file = DATA_DIR / "rfm_customers.csv"

    if not rfm_file.exists():
        st.warning("RFM results not found. Run src/rfm.py first.")
    else:
        rfm = pd.read_csv(rfm_file)

        col1, col2, col3 = st.columns(3)
        col1.metric("Customers", f"{rfm['Customer_ID'].nunique():,}")
        col2.metric("Total Customer Value", f"£{rfm['Monetary'].sum():,.0f}")
        col3.metric("Average Recency", f"{rfm['Recency'].mean():.1f} days")

        st.subheader("Customer Segments")
        st.bar_chart(rfm["Segment"].value_counts())

        st.subheader("RFM Customer Table")
        st.dataframe(
            rfm[
                [
                    "Customer_ID", "Recency", "Frequency",
                    "Monetary", "RFM_Score", "Segment",
                ]
            ].sort_values("Monetary", ascending=False),
            use_container_width=True,
        )


# --------------------------------------------------
# 6. Customer Segmentation
# --------------------------------------------------
elif page == "Customer Segmentation":
    st.header("Customer Segmentation")

    cluster_file = DATA_DIR / "customer_clusters.csv"
    profile_file = DATA_DIR / "kmeans_cluster_profile.csv"

    if not cluster_file.exists():
        st.warning(
            "Customer clusters not found. Run src/segmentation.py first."
        )
    else:
        clusters = pd.read_csv(cluster_file)

        if "Cluster" not in clusters.columns:
            st.error("The cluster output does not contain a Cluster column.")
        else:
            st.subheader("Cluster Distribution")

            cluster_counts = (
                clusters["Cluster"]
                .value_counts()
                .sort_index()
                .rename(index=lambda x: f"Cluster {x}")
            )
            st.bar_chart(cluster_counts)

            st.dataframe(
                cluster_counts.rename("Customers").reset_index(
                    names="Cluster"
                ),
                hide_index=True,
                use_container_width=True,
            )

            if profile_file.exists():
                profile = pd.read_csv(profile_file)
                st.subheader("Cluster Profiles")
                st.dataframe(profile, use_container_width=True)

            st.subheader("Customers by Cluster")
            available_features = [
                col for col in ["Recency", "Frequency", "Monetary"]
                if col in clusters.columns
            ]

            if available_features:
                selected_feature = st.selectbox(
                    "Choose a metric to compare",
                    available_features,
                )
                summary = clusters.groupby("Cluster")[selected_feature].mean()
                st.bar_chart(summary)

            st.dataframe(clusters, use_container_width=True)


# --------------------------------------------------
# 7. Churn Prediction
# --------------------------------------------------
elif page == "Churn Prediction":
    st.header("Customer Churn Prediction")

    st.write(
        "Review the model's ability to identify customers who did not "
        "purchase during the defined future observation window."
    )

    metrics_file = DATA_DIR / "churn_model_metrics.csv"
    training_file = DATA_DIR / "churn_training_data.csv"

    if not metrics_file.exists():
        st.warning(
            "Churn metrics not found. Run src/churn.py and "
            "src/churn_model.py first."
        )
    else:
        metrics = pd.read_csv(metrics_file)

        # Normalize column names.
        metrics.columns = (
            metrics.columns.str.strip().str.lower()
        )

        # Standardize common metric column names.
        metrics = metrics.rename(
            columns={
                "roc_auc_score": "roc_auc",
                "f1_score": "f1",
                "f1-score": "f1",
            }
        )

        st.subheader("Model Performance")

        st.dataframe(
            metrics,
            hide_index=True,
            use_container_width=True
        )

        required_columns = {
            "model",
            "accuracy",
            "roc_auc"
        }

        if required_columns.issubset(metrics.columns):
            best = metrics.loc[metrics["roc_auc"].idxmax()]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Best Model",
                str(best["model"])
            )

            col2.metric(
                "Accuracy",
                f'{best["accuracy"]:.2%}'
            )

            col3.metric(
                "ROC-AUC",
                f'{best["roc_auc"]:.4f}'
            )
        else:
            st.warning(
                "The metrics CSV does not contain the expected "
                "model, accuracy, and roc_auc columns."
            )

        # Historical churn labels
        if training_file.exists():
            churn_data = pd.read_csv(training_file)

            if "Churn" in churn_data.columns:
                st.subheader("Historical Churn Labels")

                churn_counts = (
                    churn_data["Churn"]
                    .map({
                        0: "Purchased in future window",
                        1: "No purchase in future window",
                    })
                    .value_counts()
                    .reindex(
                        [
                            "Purchased in future window",
                            "No purchase in future window",
                        ],
                        fill_value=0
                    )
                )

                st.bar_chart(
                    churn_counts,
                    y_label="Number of Customers"
                )

                st.caption(
                    "No purchase during a 90-day observation window "
                    "is an inactivity proxy, not proof of permanent churn."
                )
            else:
                st.warning(
                    "The Churn column was not found in the training data."
                )
        else:
            st.info(
                "Historical churn labels are unavailable. "
                "Run src/churn.py first."
            )

# --------------------------------------------------
# 8. Demand Forecasting
# --------------------------------------------------
elif page == "Demand Forecasting":
    st.header("Demand Forecasting")

    predictions_file = DATA_DIR / "forecast_evaluation_predictions.csv"
    metrics_file = DATA_DIR / "forecast_metrics.csv"

    if predictions_file.exists():
        forecast = pd.read_csv(predictions_file, parse_dates=["Date"])
        forecast = forecast.sort_values("Date").set_index("Date")

        st.subheader("Actual vs Predicted Revenue")
        available_cols = [
            col for col in [
                "Actual_Revenue", "Seasonal_Naive", "Random_Forest"
            ]
            if col in forecast.columns
        ]
        st.line_chart(forecast[available_cols])

    else:
        st.warning(
            "Forecast results not found. Run src/forecasting_model.py first."
        )

    if metrics_file.exists():
        st.subheader("Forecast Evaluation")
        st.dataframe(
            pd.read_csv(metrics_file),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Evaluation is one-day-ahead on a historical test window; "
        "it is not a recursive multi-day forecast."
    )


# --------------------------------------------------
# 9. Inventory Insights
# --------------------------------------------------
elif page == "Inventory Insights":
    st.header("Inventory Insights")
    st.write(
        "Analyse historical product demand, revenue contribution, "
        "and sales activity to support inventory planning."
    )

    # Prepare product-level summary
    inventory_data = sales.copy()

    inventory_data["StockCode"] = (
        inventory_data["StockCode"].astype(str).str.strip()
    )

    product_summary = (
        inventory_data.groupby("StockCode")
        .agg(
            UnitsSold=("Quantity", "sum"),
            Revenue=("LineValue", "sum"),
            Orders=("Invoice", "nunique"),
        )
        .reset_index()
    )

    # Add product descriptions when available
    if "Description" in inventory_data.columns:
        descriptions = (
            inventory_data.dropna(subset=["Description"])
            .groupby("StockCode")["Description"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
            .rename("Product")
            .reset_index()
        )

        product_summary = product_summary.merge(
            descriptions,
            on="StockCode",
            how="left"
        )
    else:
        product_summary["Product"] = "Description unavailable"

    product_summary["Product"] = (
        product_summary["Product"].fillna("Description unavailable")
    )

    # Sort and calculate contribution indicators
    product_summary = product_summary.sort_values(
        "UnitsSold",
        ascending=False
    )

    total_revenue = product_summary["Revenue"].sum()
    total_units = product_summary["UnitsSold"].sum()

    product_summary["RevenueShare (%)"] = (
        product_summary["Revenue"] / total_revenue * 100
        if total_revenue != 0
        else 0
    )

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Products Analysed",
        f"{len(product_summary):,}"
    )

    col2.metric(
        "Total Units Sold",
        f"{total_units:,.0f}"
    )

    col3.metric(
        "Total Revenue",
        f"£{total_revenue:,.2f}"
    )

    col4.metric(
        "Products with Sales",
        f"{(product_summary['UnitsSold'] > 0).sum():,}"
    )

    # Top products by units sold
    st.subheader("Top 10 Products by Units Sold")

    top_units = product_summary.head(10).set_index("StockCode")

    st.bar_chart(top_units["UnitsSold"], y_label="Units Sold")

    # Top products by revenue
    st.subheader("Top 10 Products by Revenue")

    top_revenue = (
        product_summary.nlargest(10, "Revenue")
        .set_index("StockCode")
    )

    st.bar_chart(top_revenue["Revenue"], y_label="Revenue (£)")

    # Product table
    st.subheader("Product Demand Table")

    search = st.text_input(
        "Search by product code or description",
        placeholder="Enter a product code or keyword..."
    )

    filtered_products = product_summary.copy()

    if search.strip():
        search_term = search.strip().lower()

        filtered_products = filtered_products[
            filtered_products["StockCode"].str.lower().str.contains(
                search_term, na=False
            )
            | filtered_products["Product"].str.lower().str.contains(
                search_term, na=False
            )
        ]

    display_columns = [
        "StockCode",
        "Product",
        "UnitsSold",
        "Revenue",
        "Orders",
        "RevenueShare (%)",
    ]

    st.dataframe(
        filtered_products[display_columns].round(2),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Inventory indicators are based on historical sales. "
        "The dataset does not provide current stock-on-hand, "
        "supplier lead times, or replenishment records, so exact "
        "reorder quantities cannot be calculated."
    )