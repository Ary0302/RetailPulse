
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "rfm_customers.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "customer_clusters.csv"


def main():
    print("Loading RFM customer data...")

    df = pd.read_csv(INPUT_FILE)

    features = ["Recency", "Frequency", "Monetary"]
    X = df[features].copy()

    # RFM values are often highly skewed.
    # log1p reduces skew while preserving zero values if present.
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])

    # Standardize features so large monetary values do not dominate.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compare candidate cluster counts.
    results = []

    print("\nEvaluating cluster counts...")
    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": score,
        })

        print(
            f"k={k} | inertia={model.inertia_:.2f} "
            f"| silhouette={score:.4f}"
        )

    results_df = pd.DataFrame(results)

    # Select the candidate with the highest silhouette score.
    best_k = int(
        results_df.loc[
            results_df["silhouette_score"].idxmax(), "k"
        ]
    )

    print(f"\nSelected cluster count: {best_k}")

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )
    df["Cluster"] = final_model.fit_predict(X_scaled)

    # Profile clusters using original, interpretable RFM values.
    profile = df.groupby("Cluster").agg(
        Customer_Count=("Customer_ID", "nunique"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Monetary=("Monetary", "sum"),
    ).round(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    results_df.to_csv(
        OUTPUT_DIR / "kmeans_evaluation.csv", index=False
    )
    profile.to_csv(
        OUTPUT_DIR / "kmeans_cluster_profile.csv"
    )

    print("\nCluster sizes:")
    print(df["Cluster"].value_counts().sort_index().to_string())

    print("\nCluster profiles (original RFM units):")
    print(profile.to_string())

    print(f"\nSaved clustered customers to: {OUTPUT_FILE}")
    print(
        "Saved evaluation to: "
        f"{OUTPUT_DIR / 'kmeans_evaluation.csv'}"
    )
    print(
        "Saved cluster profiles to: "
        f"{OUTPUT_DIR / 'kmeans_cluster_profile.csv'}"
    )


if __name__ == "__main__":
    main()
