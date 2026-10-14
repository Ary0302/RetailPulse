# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting Platform

RetailPulse is a data science and analytics project that analyzes retail transaction data to uncover customer behavior, segment customers, identify potential churn, and forecast daily revenue. It provides an interactive Streamlit dashboard for exploring sales, customer analytics, model evaluation, and product-level performance.

## Key Features

- **Sales Analytics:** Explore revenue, transaction trends, countries, and products.
- **RFM Analysis:** Analyze customers using Recency, Frequency, and Monetary value.
- **Customer Segmentation:** Apply K-Means clustering to group customers based on purchasing behavior.
- **Churn Prediction:** Use Logistic Regression and Random Forest to classify customer inactivity.
- **Demand Forecasting:** Compare Random Forest revenue forecasts against a seasonal-naive baseline.
- **Inventory Insights:** Examine historical product sales and revenue to support inventory analysis.
- **Interactive Dashboard:** Filter results, explore charts, and download supported tables as CSV files.

## Technology Stack

- **Language:** Python
- **Data Analysis:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Model Persistence:** joblib
- **Visualization and Dashboard:** Streamlit
- **Development Tools:** VS Code, Jupyter Notebook, Git, GitHub


## Project Architecture

```text
RetailPulse/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_cleaned_data_eda.ipynb
│   └── 03_demand_forecasting.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── rfm.py
│   ├── segmentation.py
│   ├── churn.py
│   ├── churn_model.py
│   ├── forecasting.py
│   └── forecasting_model.py
├── requirements.txt
└── README.md
```

## Dataset

RetailPulse uses the Online Retail II dataset, containing historical transactions from December 2009 to December 2011.

The processed sales dataset contains 1,007,913 rows. Of these, 228,488 rows (22.67%) have missing customer IDs. These transactions can contribute to sales analytics but cannot reliably support individual customer analysis.

## Model Results

### Customer Churn

Logistic Regression was selected over Random Forest based on ROC-AUC.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 69.76% | 66.51% |
| Precision | 66.87% | 64.72% |
| Recall | 76.47% | 70.35% |
| F1-score | 71.35% | 67.42% |
| ROC-AUC | 0.7646 | 0.7350 |

The positive class represents customers who made no purchase during the subsequent 90-day observation window.

### Demand Forecasting

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal Naive | £14,104.06 | £25,118.05 |
| Random Forest | £12,537.54 | £23,893.58 |

Random Forest reduced MAE by approximately 11.1% and RMSE by approximately 4.9% on the existing 60-day chronological evaluation.

The forecast model uses lagged revenue, rolling averages, weekday, and month features. The evaluation uses actual historical lag values; multi-day recursive forecasting has not been validated.

## Dashboard

The Streamlit dashboard includes:

1. Sales Overview
2. RFM Analytics
3. Customer Segmentation
4. Churn Prediction
5. Demand Forecasting
6. Inventory Insights

It supports interactive filters, visual analytics, and CSV downloads in applicable sections.

Inventory Insights analyzes historical product sales. Exact stock-on-hand and supplier lead-time information is unavailable, so the project does not claim live reorder quantities.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ary0302/RetailPulse.git
cd RetailPulse
```

### 2. Create a virtual environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the Online Retail II workbook at:

```text
data/raw/online_retail_II.xlsx
```

The raw workbook is not committed to GitHub. Obtain the dataset separately before running the data preparation pipeline.

### 5. Launch the dashboard

```cmd
streamlit run app/app.py
```

The dashboard requires the processed data and trained model artifacts at the paths configured in the project. A fresh clone may require running the data preparation and modeling scripts first.

## Limitations and Future Improvements

- Incorporate recent retail transaction data.
- Investigate the impact of missing customer IDs.
- Evaluate churn decision thresholds and business costs.
- Test forecasting across multiple chronological holdout periods.
- Validate recursive multi-day forecasting.
- Incorporate stock-on-hand and supplier lead-time data for reorder recommendations.
- Add automated tests and robust handling of missing files.

## Project Status

The RetailPulse MVP includes data cleaning, RFM analysis, K-Means segmentation, churn modeling, demand forecasting, and an interactive Streamlit dashboard. The application has passed initial functional checks.

Results are based on historical data ending December 9, 2011, and should not be interpreted as live retail forecasts.
