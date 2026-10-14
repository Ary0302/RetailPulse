
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
SALES_FILE = DATA_DIR / "retail_sales.csv.gz"

st.set_page_config(
    page_title="RetailPulse | Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROFESSIONAL DARK-NAVY THEME
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --rp-bg: #0b1120;
        --rp-panel: #111c30;
        --rp-border: #24344d;
        --rp-text: #e8eef8;
        --rp-muted: #94a8c4;
        --rp-accent: #55b7ff;
        --rp-green: #36d6a0;
    }

    .stApp {
        background-color: var(--rp-bg);
        color: var(--rp-text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #101b30 0%, #0b1120 100%
        );
        border-right: 1px solid var(--rp-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--rp-text);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #f2f6ff !important;
        letter-spacing: -0.4px;
    }

    p, label, .stMarkdown {
        color: var(--rp-text);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            135deg, #14243b 0%, #101a2b 100%
        );
        border: 1px solid var(--rp-border);
        padding: 18px 20px;
        border-radius: 14px;
        min-height: 115px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--rp-muted) !important;
        font-size: 0.88rem;
    }

    [data-testid="stMetricValue"] {
        color: #f4f8ff !important;
        font-weight: 700;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--rp-border);
        border-radius: 14px;
    }

    .rp-eyebrow {
        color: var(--rp-accent);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .rp-subtitle {
        color: var(--rp-muted);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .rp-note {
        color: var(--rp-muted);
        font-size: 0.82rem;
        padding-top: 8px;
    }

    div.stButton > button {
        border-radius: 9px;
        border: 1px solid var(--rp-border);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--rp-muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data(show_spinner="Loading retail transactions...")
def load_sales(file_path, modified_time):
    df = pd.read_csv(
        file_path,
        parse_dates=["InvoiceDate"],
        dtype={
            "Invoice": "string",
            "StockCode": "string",
        },
        low_memory=False,
    )

    df["LineValue"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    if "Customer_ID" not in df.columns:
        df["Customer_ID"] = pd.NA

    if "Country" not in df.columns:
        df["Country"] = "Unknown"

    if "Description" not in df.columns:
        df["Description"] = "Description unavailable"

    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["Country"] = df["Country"].fillna("Unknown").astype(str)
    df["Description"] = df["Description"].fillna(
        "Description unavailable"
    ).astype(str)

    return df


@st.cache_data(show_spinner=False)
def load_csv(file_path, modified_time):
    return pd.read_csv(file_path, low_memory=False)


def show_missing_file(file_path, script_name):
    st.warning(f"Required file not found: `{file_path.name}`")
    st.code(f"python {script_name}")
    st.caption(
        "Run the indicated script from the RetailPulse project root "
        "if the output has not been generated."
    )


def section_header(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def money(value):
    return f"£{value:,.2f}"


def filter_sales(df, key_prefix):
    """Reusable date and country filters."""
    filtered = df.copy()

    if filtered.empty:
        return filtered

    with st.expander("⚙️ Filters", expanded=True):
        col1, col2 = st.columns([1.5, 1])

        valid_dates = filtered["InvoiceDate"].dropna()

        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            with col1:
                date_range = st.date_input(
                    "Transaction date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{key_prefix}_dates",
                )

            with col2:
                countries = sorted(
                    filtered["Country"].dropna().unique().tolist()
                )
                selected_countries = st.multiselect(
                    "Countries",
                    options=countries,
                    default=countries,
                    key=f"{key_prefix}_countries",
                )

            if isinstance(date_range, (tuple, list)):
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered = filtered[
                        filtered["InvoiceDate"].dt.date.between(
                            start_date, end_date
                        )
                    ]
                elif len(date_range) == 1:
                    filtered = filtered[
                        filtered["InvoiceDate"].dt.date
                        == date_range[0]
                    ]

            filtered = filtered[
                filtered["Country"].isin(selected_countries)
            ]

    return filtered


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="font-size: 2rem;">📊</div>
            <div style="
                font-size: 1.65rem;
                font-weight: 800;
                letter-spacing: -0.8px;
            ">RetailPulse</div>
            <div style="
                color: #94a8c4;
                font-size: 0.82rem;
            ">AI-Powered Retail Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Sales Overview",
            "RFM Analytics",
            "Customer Segmentation",
            "Churn Prediction",
            "Demand Forecasting",
            "Inventory Insights",
        ],
        key="rp_navigation",
    )

    st.divider()

    st.markdown("**DATA COVERAGE**")

    st.caption("Historical Online Retail II dataset")

    if SALES_FILE.exists():
        st.caption(
            f"Last modified: "
            f"{pd.Timestamp(SALES_FILE.stat().st_mtime, unit='s'):%d %b %Y}"
        )

    st.markdown(
        """
        <div class="rp-note">
            RetailPulse MVP<br>
            Sales • Customers • Forecasting
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5. GLOBAL HEADER AND SALES DATA
# =========================================================

st.markdown(
    '<div class="rp-eyebrow">RETAIL ANALYTICS PLATFORM</div>',
    unsafe_allow_html=True,
)

st.title("RetailPulse")

st.markdown(
    '<div class="rp-subtitle">'
    'Understand sales performance, customer behaviour, churn patterns, '
    'and historical product demand.'
    '</div>',
    unsafe_allow_html=True,
)

if not SALES_FILE.exists():
    st.error(f"Sales dataset not found: {SALES_FILE}")
    st.info("Run the data-cleaning pipeline first.")
    st.stop()

sales = load_sales(
    str(SALES_FILE),
    SALES_FILE.stat().st_mtime,
)


# =========================================================
# 6. SALES OVERVIEW
# =========================================================

if page == "Sales Overview":

    st.header("Sales Overview")
    st.caption(
        "Explore retail performance using date and country filters."
    )

    filtered_sales = filter_sales(sales, "sales_overview")

    if filtered_sales.empty:
        st.warning("No transactions match the selected filters.")
        st.stop()

    total_revenue = filtered_sales["LineValue"].sum()
    total_units = filtered_sales["Quantity"].sum()
    total_orders = filtered_sales["Invoice"].nunique()
    total_products = filtered_sales["StockCode"].nunique()
    total_customers = filtered_sales["Customer_ID"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Revenue", money(total_revenue))
    c2.metric("Units Sold", f"{total_units:,.0f}")
    c3.metric("Orders", f"{total_orders:,}")
    c4.metric("Products", f"{total_products:,}")

    c1, c2 = st.columns(2)

    c1.metric("Identified Customers", f"{total_customers:,}")
    c2.metric(
        "Average Revenue per Order",
        money(
            total_revenue / total_orders
            if total_orders else 0
        ),
    )

    st.divider()

    monthly = (
        filtered_sales.dropna(subset=["InvoiceDate"])
        .set_index("InvoiceDate")["LineValue"]
        .resample("MS")
        .sum()
    )

    monthly.index = monthly.index.strftime("%b %Y")

    left, right = st.columns([1.6, 1])

    with left:
        section_header("Revenue Trend", "Monthly sales revenue")
        st.line_chart(
            monthly,
            y_label="Revenue (£)",
            color="#55b7ff",
        )

    with right:
        section_header("Country Contribution", "Top 10 countries by revenue")

        country_revenue = (
            filtered_sales.groupby("Country")["LineValue"]
            .sum()
            .nlargest(10)
            .sort_values()
        )

        st.bar_chart(
            country_revenue,
            x_label="Revenue (£)",
        )

    left, right = st.columns(2)

    with left:
        section_header("Top Products", "Ranked by units sold")

        top_n = st.selectbox(
            "Number of products",
            [5, 10, 15, 20],
            index=1,
            key="sales_top_n",
        )

        top_products = (
            filtered_sales.groupby("StockCode")
            .agg(
                UnitsSold=("Quantity", "sum"),
                Description=("Description", "first"),
            )
            .nlargest(top_n, "UnitsSold")
            .sort_values("UnitsSold")
        )

        top_products["Product"] = (
            top_products["Description"]
            .fillna(top_products.index.to_series())
            .astype(str)
            .str.slice(0, 40)
        )

        st.bar_chart(
            top_products.set_index("Product")["UnitsSold"],
            x_label="Units sold",
        )

    with right:
        section_header("Recent Transactions", "Latest transaction records")

        recent = filtered_sales.sort_values(
            "InvoiceDate", ascending=False
        ).head(15)

        st.dataframe(
            recent[
                [
                    "InvoiceDate",
                    "Invoice",
                    "StockCode",
                    "Country",
                    "Quantity",
                    "LineValue",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Revenue is transaction value, not profit. Customer counts exclude "
        "transactions with missing customer IDs."
    )


# =========================================================
# 7. RFM ANALYTICS
# =========================================================

elif page == "RFM Analytics":

    st.header("RFM Customer Analytics")
    st.caption(
        "Segment customers by Recency, Frequency, and Monetary value."
    )

    rfm_file = DATA_DIR / "rfm_customers.csv"

    if not rfm_file.exists():
        show_missing_file(rfm_file, "src/rfm.py")
        st.stop()

    rfm = load_csv(
        str(rfm_file),
        rfm_file.stat().st_mtime,
    )

    required = {
        "Customer_ID", "Recency", "Frequency", "Monetary", "Segment"
    }

    if not required.issubset(rfm.columns):
        st.error("The RFM file is missing required columns.")
        st.stop()

    segments = sorted(rfm["Segment"].dropna().unique().tolist())

    selected_segments = st.multiselect(
        "Filter customer segments",
        segments,
        default=segments,
        key="rfm_segments",
    )

    rfm_filtered = rfm[rfm["Segment"].isin(selected_segments)]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", f"{rfm_filtered['Customer_ID'].nunique():,}")
    c2.metric("Total Monetary Value", money(rfm_filtered["Monetary"].sum()))
    c3.metric(
        "Average Recency",
        f"{rfm_filtered['Recency'].mean():.1f} days"
        if not rfm_filtered.empty else "—",
    )
    c4.metric(
        "Average Frequency",
        f"{rfm_filtered['Frequency'].mean():.1f}"
        if not rfm_filtered.empty else "—",
    )

    if rfm_filtered.empty:
        st.info("Select at least one customer segment.")
        st.stop()

    left, right = st.columns(2)

    with left:
        section_header("Segment Distribution")
        st.bar_chart(rfm_filtered["Segment"].value_counts())

    with right:
        section_header("Monetary Value by Segment")
        monetary_by_segment = (
            rfm_filtered.groupby("Segment")["Monetary"]
            .sum()
            .sort_values()
        )
        st.bar_chart(monetary_by_segment, x_label="Monetary value (£)")

    st.subheader("Customer Explorer")

    min_value = float(rfm_filtered["Monetary"].min())
    max_value = float(rfm_filtered["Monetary"].max())

    if min_value < max_value:
        monetary_range = st.slider(
            "Customer monetary value range (£)",
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
            key="rfm_monetary_range",
        )

        rfm_filtered = rfm_filtered[
            rfm_filtered["Monetary"].between(*monetary_range)
        ]

    search_customer = st.text_input(
        "Search Customer ID",
        key="rfm_customer_search",
    )

    if search_customer.strip():
        rfm_filtered = rfm_filtered[
            rfm_filtered["Customer_ID"].astype(str)
            .str.contains(search_customer.strip(), na=False)
        ]

    sort_by = st.selectbox(
        "Sort customers by",
        ["Monetary", "Recency", "Frequency"],
        key="rfm_sort",
    )

    st.dataframe(
        rfm_filtered.sort_values(sort_by, ascending=False),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download filtered RFM data",
        data=rfm_filtered.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_rfm_filtered.csv",
        mime="text/csv",
    )


# =========================================================
# 8. CUSTOMER SEGMENTATION
# =========================================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")
    st.caption("Explore K-Means clusters and compare customer behaviour.")

    cluster_file = DATA_DIR / "customer_clusters.csv"
    profile_file = DATA_DIR / "kmeans_cluster_profile.csv"

    if not cluster_file.exists():
        show_missing_file(cluster_file, "src/segmentation.py")
        st.stop()

    clusters = load_csv(
        str(cluster_file),
        cluster_file.stat().st_mtime,
    )

    if "Cluster" not in clusters.columns:
        st.error("The cluster output does not contain a Cluster column.")
        st.stop()

    cluster_ids = sorted(clusters["Cluster"].dropna().unique().tolist())

    selected_clusters = st.multiselect(
        "Select clusters",
        cluster_ids,
        default=cluster_ids,
        format_func=lambda x: f"Cluster {x}",
        key="cluster_filter",
    )

    cluster_filtered = clusters[
        clusters["Cluster"].isin(selected_clusters)
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric("Customers Displayed", f"{len(cluster_filtered):,}")
    c2.metric("Clusters Selected", f"{len(selected_clusters)}")

    if "Monetary" in cluster_filtered.columns:
        c3.metric(
            "Total Monetary Value",
            money(cluster_filtered["Monetary"].sum()),
        )
    else:
        c3.metric("Customer Records", f"{len(cluster_filtered):,}")

    left, right = st.columns(2)

    with left:
        section_header("Cluster Distribution")
        st.bar_chart(
            cluster_filtered["Cluster"]
            .value_counts()
            .sort_index()
            .rename(index=lambda x: f"Cluster {x}"),
        )

    with right:
        features = [
            col for col in ["Recency", "Frequency", "Monetary"]
            if col in cluster_filtered.columns
        ]

        if features:
            selected_feature = st.selectbox(
                "Compare average customer metric",
                features,
                key="cluster_feature",
            )

            profile = (
                cluster_filtered.groupby("Cluster")[selected_feature]
                .mean()
                .sort_index()
            )

            st.bar_chart(profile, y_label=selected_feature)

    if profile_file.exists():
        st.subheader("Cluster Profiles")

        cluster_profile = load_csv(
            str(profile_file),
            profile_file.stat().st_mtime,
        )

        st.dataframe(
            cluster_profile,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Customer Records")

    st.dataframe(
        cluster_filtered,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download cluster records",
        data=cluster_filtered.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_customer_clusters.csv",
        mime="text/csv",
    )


# =========================================================
# 9. CHURN PREDICTION
# =========================================================

elif page == "Churn Prediction":

    st.header("Customer Churn Prediction")
    st.caption(
        "Historical model evaluation and customer inactivity analysis."
    )

    metrics_file = DATA_DIR / "churn_model_metrics.csv"
    training_file = DATA_DIR / "churn_training_data.csv"

    if not metrics_file.exists():
        show_missing_file(metrics_file, "src/churn_model.py")
        st.stop()

    metrics = load_csv(
        str(metrics_file),
        metrics_file.stat().st_mtime,
    )

    metrics.columns = metrics.columns.str.strip().str.lower()

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
        use_container_width=True,
    )

    if {"model", "accuracy", "roc_auc"}.issubset(metrics.columns):
        best = metrics.loc[metrics["roc_auc"].idxmax()]

        c1, c2, c3 = st.columns(3)

        c1.metric("Best Model", str(best["model"]))
        c2.metric("Accuracy", f'{best["accuracy"]:.2%}')
        c3.metric("ROC-AUC", f'{best["roc_auc"]:.4f}')

    if training_file.exists():
        churn_data = load_csv(
            str(training_file),
            training_file.stat().st_mtime,
        )

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
                    fill_value=0,
                )
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Purchased in Future Window",
                f"{int(churn_counts.iloc[0]):,}",
            )

            c2.metric(
                "No Purchase in Future Window",
                f"{int(churn_counts.iloc[1]):,}",
            )

            st.bar_chart(
                churn_counts,
                y_label="Customers",
            )

            if "Customer_ID" in churn_data.columns:
                st.subheader("Historical Customer Explorer")

                churn_options = {
                    0: "Purchased in future window",
                    1: "No purchase in future window",
                }

                selected_label = st.selectbox(
                    "Filter historical labels",
                    list(churn_options.keys()),
                    format_func=lambda x: churn_options[x],
                    key="churn_label_filter",
                )

                churn_filtered = churn_data[
                    churn_data["Churn"] == selected_label
                ]

                st.dataframe(
                    churn_filtered,
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "Download churn records",
                    data=churn_filtered.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="retailpulse_churn_records.csv",
                    mime="text/csv",
                )

    st.caption(
        "The target is no purchase during a defined 90-day observation "
        "window. This is an inactivity proxy, not proof of permanent churn. "
        "The page displays historical evaluation, not live predictions."
    )


# =========================================================
# 10. DEMAND FORECASTING
# =========================================================

elif page == "Demand Forecasting":

    st.header("Demand Forecasting")
    st.caption(
        "Compare historical revenue with one-day-ahead model predictions."
    )

    predictions_file = DATA_DIR / "forecast_evaluation_predictions.csv"
    metrics_file = DATA_DIR / "forecast_metrics.csv"

    if predictions_file.exists():
        forecast = load_csv(
            str(predictions_file),
            predictions_file.stat().st_mtime,
        )

        if "Date" in forecast.columns:
            forecast["Date"] = pd.to_datetime(
                forecast["Date"], errors="coerce"
            )
            forecast = forecast.dropna(subset=["Date"])
            forecast = forecast.sort_values("Date")

            if not forecast.empty:
                min_date = forecast["Date"].min().date()
                max_date = forecast["Date"].max().date()

                date_range = st.date_input(
                    "Forecast evaluation date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="forecast_dates",
                )

                if isinstance(date_range, (tuple, list)):
                    if len(date_range) == 2:
                        forecast = forecast[
                            forecast["Date"].dt.date.between(
                                date_range[0], date_range[1]
                            )
                        ]

                chart_columns = [
                    col for col in [
                        "Actual_Revenue",
                        "Random_Forest",
                        "Seasonal_Naive",
                    ]
                    if col in forecast.columns
                ]

                if chart_columns and not forecast.empty:
                    st.subheader("Actual vs Predicted Revenue")

                    chart_data = forecast.set_index("Date")[chart_columns]

                    st.line_chart(
                        chart_data,
                        y_label="Revenue (£)",
                    )

                    if "Actual_Revenue" in forecast.columns:
                        st.subheader("Evaluation Period Summary")

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Days Displayed",
                            f"{len(forecast):,}",
                        )

                        c2.metric(
                            "Actual Revenue",
                            money(forecast["Actual_Revenue"].sum()),
                        )

                        if "Random_Forest" in forecast.columns:
                            c3.metric(
                                "Predicted Revenue",
                                money(forecast["Random_Forest"].sum()),
                            )

                    st.download_button(
                        "Download forecast evaluation data",
                        data=forecast.to_csv(index=False).encode("utf-8"),
                        file_name="retailpulse_forecast_evaluation.csv",
                        mime="text/csv",
                    )
                elif forecast.empty:
                    st.info("No forecast records match the selected dates.")
                else:
                    st.warning("No recognized forecast columns were found.")
        else:
            st.warning("Forecast output is missing the Date column.")
    else:
        show_missing_file(
            predictions_file,
            "src/forecasting_model.py",
        )

    if metrics_file.exists():
        st.subheader("Forecast Evaluation Metrics")

        forecast_metrics = load_csv(
            str(metrics_file),
            metrics_file.stat().st_mtime,
        )

        st.dataframe(
            forecast_metrics,
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "This evaluation uses a historical test window and one-day-ahead "
        "predictions. It is not a recursive multi-day forecast. "
        "The source dataset ends in December 2011."
    )


# =========================================================
# 11. INVENTORY INSIGHTS
# =========================================================

elif page == "Inventory Insights":

    st.header("Inventory Insights")
    st.caption(
        "Analyse historical product demand and revenue contribution."
    )

    inventory_data = filter_sales(sales, "inventory")

    if inventory_data.empty:
        st.warning("No transactions match the selected filters.")
        st.stop()

    inventory_data["StockCode"] = (
        inventory_data["StockCode"].astype(str).str.strip()
    )

    product_summary = (
        inventory_data.groupby("StockCode")
        .agg(
            UnitsSold=("Quantity", "sum"),
            Revenue=("LineValue", "sum"),
            Orders=("Invoice", "nunique"),
            Product=("Description", "first"),
        )
        .reset_index()
    )

    product_summary["Product"] = product_summary["Product"].fillna(
        "Description unavailable"
    )

    total_revenue = product_summary["Revenue"].sum()
    total_units = product_summary["UnitsSold"].sum()

    product_summary["RevenueShare (%)"] = (
        product_summary["Revenue"] / total_revenue * 100
        if total_revenue else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Products Analysed", f"{len(product_summary):,}")
    c2.metric("Units Sold", f"{total_units:,.0f}")
    c3.metric("Revenue", money(total_revenue))
    c4.metric(
        "Products with Positive Sales",
        f"{(product_summary['UnitsSold'] > 0).sum():,}",
    )

    st.divider()

    top_n = st.selectbox(
        "Number of products to display",
        [5, 10, 15, 20],
        index=1,
        key="inventory_top_n",
    )

    left, right = st.columns(2)

    with left:
        section_header("Top Products by Units Sold")

        top_units = (
            product_summary.nlargest(top_n, "UnitsSold")
            .sort_values("UnitsSold")
            .set_index("StockCode")
        )

        st.bar_chart(
            top_units["UnitsSold"],
            x_label="Units sold",
        )

    with right:
        section_header("Top Products by Revenue")

        top_revenue = (
            product_summary.nlargest(top_n, "Revenue")
            .sort_values("Revenue")
            .set_index("StockCode")
        )

        st.bar_chart(
            top_revenue["Revenue"],
            x_label="Revenue (£)",
        )

    st.subheader("Product Demand Table")

    search = st.text_input(
        "Search by product code or description",
        placeholder="Enter a product code or keyword...",
        key="inventory_search",
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

    sort_option = st.selectbox(
        "Sort product table by",
        ["UnitsSold", "Revenue", "Orders", "RevenueShare (%)"],
        key="inventory_sort",
    )

    filtered_products = filtered_products.sort_values(
        sort_option,
        ascending=False,
    )

    st.dataframe(
        filtered_products[
            [
                "StockCode",
                "Product",
                "UnitsSold",
                "Revenue",
                "Orders",
                "RevenueShare (%)",
            ]
        ].round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download product demand report",
        data=filtered_products.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_inventory_report.csv",
        mime="text/csv",
    )

    st.caption(
        "Orders are distinct invoices containing each product. "
        "This dataset does not provide current stock-on-hand or supplier "
        "lead times, so exact reorder quantities cannot be calculated."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        color: #94a8c4;
        font-size: 0.8rem;
        text-align: center;
        padding: 8px;
    ">
        RetailPulse • AI-Powered Customer Analytics & Demand Forecasting
        <br>
        Historical analysis — not live retail data
    </div>
    """,
    unsafe_allow_html=True,
)