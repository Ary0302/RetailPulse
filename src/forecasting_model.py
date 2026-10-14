
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "daily_sales.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TARGET = "Revenue"
TEST_DAYS = 60


def make_features(series):
    """Create features using only past revenue observations."""
    frame = pd.DataFrame({"Revenue": series})

    frame["lag_1"] = frame["Revenue"].shift(1)
    frame["lag_7"] = frame["Revenue"].shift(7)
    frame["lag_14"] = frame["Revenue"].shift(14)
    frame["lag_28"] = frame["Revenue"].shift(28)

    frame["rolling_mean_7"] = (
        frame["Revenue"].shift(1).rolling(7).mean()
    )
    frame["rolling_mean_28"] = (
        frame["Revenue"].shift(1).rolling(28).mean()
    )

    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month

    return frame.dropna()


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Daily sales file not found. Run src/forecasting.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    daily = daily.set_index("Date").sort_index()

    series = daily[TARGET].astype(float).asfreq("D", fill_value=0)

    features = make_features(series)

    # Chronological split: the final 60 days are held out.
    split_date = series.index[-TEST_DAYS]

    train = features[features.index < split_date]
    test = features[features.index >= split_date]

    X_train = train.drop(columns=["Revenue"])
    y_train = train["Revenue"]
    X_test = test.drop(columns=["Revenue"])
    y_test = test["Revenue"]

    # Baseline: same weekday in the previous week.
    baseline_pred = series.shift(7).loc[test.index]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    results = pd.DataFrame({
        "Actual_Revenue": y_test,
        "Seasonal_Naive": baseline_pred,
        "Random_Forest": predictions
    })
    results.to_csv(
        OUTPUT_DIR / "forecast_evaluation_predictions.csv"
    )

    metrics = []
    for name, predicted in [
        ("Seasonal Naive", baseline_pred),
        ("Random Forest", predictions)
    ]:
        metrics.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predicted),
            "RMSE": mean_squared_error(y_test, predicted) ** 0.5
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        OUTPUT_DIR / "forecast_metrics.csv", index=False
    )

    # Retrain the final model on all available historical features.
    final_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(
        features.drop(columns=["Revenue"]),
        features["Revenue"]
    )

    joblib.dump(final_model, MODEL_DIR / "revenue_forecast_model.joblib")

    print("Forecast model evaluation completed.")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Test period: {test.index.min().date()} to {test.index.max().date()}")
    print("\nEvaluation metrics (lower is better):")
    print(metrics_df.to_string(index=False))
    print("\nSaved predictions: data/processed/forecast_evaluation_predictions.csv")
    print("Saved metrics: data/processed/forecast_metrics.csv")
    print("Saved model: models/revenue_forecast_model.joblib")


if __name__ == "__main__":
    main()

