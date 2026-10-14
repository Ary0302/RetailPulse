
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "churn_training_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "UniqueProducts",
    "TotalQuantity",
    "AvgOrderValue",
]
TARGET = "Churn"


def main():
    print("Loading churn training dataset...")
    df = pd.read_csv(INPUT_FILE)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Split before fitting scalers or models.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
        }
        results.append(metrics)
        trained_models[name] = model

        print("\nMetrics:")
        for metric, value in metrics.items():
            if metric != "Model":
                print(f"{metric}: {value:.4f}")

        print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
        print(confusion_matrix(y_test, predictions))

        print("\nClassification report:")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

    results_df = pd.DataFrame(results).sort_values(
        "ROC_AUC", ascending=False
    )

    # Select by ROC-AUC, not accuracy alone.
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")
    results_df.to_csv(
        OUTPUT_DIR / "churn_model_metrics.csv", index=False
    )

    summary = {
        "best_model_by_roc_auc": best_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "metrics": results,
        "note": (
            "Churn means no purchase in the defined 90-day future window. "
            "Metrics measure this historical inactivity label, not "
            "guaranteed future customer loss."
        ),
    }

    with open(
        OUTPUT_DIR / "churn_model_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'churn_model.joblib'}")
    print(
        "Saved metrics: "
        f"{OUTPUT_DIR / 'churn_model_metrics.csv'}"
    )
    print(
        "Saved report: "
        f"{OUTPUT_DIR / 'churn_model_report.json'}"
    )


if __name__ == "__main__":
    main()
