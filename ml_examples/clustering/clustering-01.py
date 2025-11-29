
"""
Clustering Pipeline (KMeans)

Pipeline Features:
- Loads scikit-learn's Digits dataset (8x8 images -> 64 features).
- Uses stratified sampling based on true digit labels (only for splitting).
- Trains KMeans (n_clusters=10) on standardized features (unsupervised).
- Evaluates clustering: Silhouette, Calinski-Harabasz, Davies-Bouldin, cluster sizes.
- Visualizes test clusters via PCA 2D scatter.
- Saves model artifacts and plots.
- Builds surrogate RandomForest to predict KMeans cluster labels and explains with SHAP:
  - Global bar + beeswarm.
  - Single prediction force plot (HTML).
- Prints textual explanation for top 3 contributing features for a single test point.
"""

import os
os.environ["TQDM_DISABLE"] = "1"  # keep console clean from progress bars

from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.ensemble import RandomForestClassifier

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Stage 1: Load Dataset ----------------
def load_digits_df():
    digits = load_digits()
    X = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])])
    y = pd.Series(digits.target, name="label")
    df = X.copy()
    df["label"] = y
    return df, list(X.columns)

if __name__ == "__main__":
    try:
        # Load dataset
        df, feature_names = load_digits_df()
        print(f"Digits dataset loaded. Rows={len(df)}, Shape={df.shape}")

        # ---------------- Stage 2: Stratified Sampling ----------------
        X = df[feature_names].copy()
        y = df["label"].copy()  # used only for stratified split; dropped before clustering

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.7, random_state=42, stratify=y
        )
        print(f"Training size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

        # Standardize features (important for KMeans)
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            columns=feature_names,
            index=X_train_raw.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw),
            columns=feature_names,
            index=X_test_raw.index
        )

        # ---------------- Stage 3: Train KMeans Model ----------------
        n_clusters = 10  # digits 0-9
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        kmeans.fit(X_train)

        # Predict clusters
        train_clusters = kmeans.predict(X_train)
        test_clusters = kmeans.predict(X_test)

        # ---------------- Stage 4: Evaluate Clustering ----------------
        sil = silhouette_score(X_test, test_clusters)
        ch = calinski_harabasz_score(X_test, test_clusters)
        db = davies_bouldin_score(X_test, test_clusters)

        print(f"Silhouette: {sil:.4f}, Calinski-Harabasz: {ch:.2f}, Davies-Bouldin: {db:.4f}")
        unique, counts = np.unique(test_clusters, return_counts=True)
        print("Cluster sizes (test set):", dict(zip(unique, counts)))

        # ---------------- Stage 5: Save Model ----------------
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        with open("./trained_models/kmeans_digits.pkl", "wb") as f:
            pickle.dump({"kmeans": kmeans, "scaler": scaler, "feature_names": feature_names}, f)

        # ---------------- Stage 6: Visualize Clusters (PCA 2D) ----------------
        Path("./explanations/kmeans_digits/").mkdir(parents=True, exist_ok=True)

        pca = PCA(n_components=2, random_state=42)
        X_test_pca = pca.fit_transform(X_test)
        df_plot = pd.DataFrame(X_test_pca, columns=["PC1", "PC2"])
        df_plot["cluster"] = test_clusters

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_plot,
            x="PC1", y="PC2",
            hue="cluster", palette="tab10", s=25, alpha=0.85, edgecolor="none"
        )
        plt.title("KMeans Clusters (Digits Test Set, PCA 2D)")
        plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("./explanations/kmeans_digits/kmeans_pca_clusters.png", dpi=160)
        plt.close()

        # Save cluster centers (inverse-transformed to original pixel scale)
        centers_scaled = kmeans.cluster_centers_
        centers = pd.DataFrame(
            scaler.inverse_transform(centers_scaled),
            columns=feature_names
        )
        centers.to_csv("./explanations/kmeans_digits/kmeans_cluster_centers.csv", index=False)
        print("Saved cluster centers (original feature scale) to CSV.")

        # ---------------- Stage 7: Surrogate Model + SHAP Explanations ----------------
        rf = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        # Train RF only on the intended features
        rf.fit(X_train[feature_names], train_clusters)

        # SHAP explainer for tree models
        background = X_train[feature_names].sample(n=min(200, len(X_train)), random_state=42)
        explainer = shap.TreeExplainer(rf, data=background)
        shap_values = explainer.shap_values(X_test[feature_names], check_additivity=False)

        # Handle multi-class SHAP outputs
        if isinstance(shap_values, list):
            # shap_values: list of arrays (n_classes, n_samples, n_features)
            # Global mean |SHAP| across classes
            shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            shap_values_for_beeswarm = shap_values[0]  # choose first class for beeswarm viz
        else:
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
            shap_values_for_beeswarm = shap_values

        # Flatten and ALIGN lengths robustly
        shap_abs_mean = np.asarray(shap_abs_mean).flatten()
        rf_features = list(X_train[feature_names].columns)  # authoritative source for feature order

        # Sanity logs (helps debug shape mismatches)
        print(f"[DEBUG] len(feature_names)={len(feature_names)}, "
              f"len(rf_features)={len(rf_features)}, "
              f"len(shap_abs_mean)={len(shap_abs_mean)}")

        # If mismatch, align using min length
        min_len = min(len(rf_features), len(shap_abs_mean))
        aligned_features = rf_features[:min_len]
        aligned_shap = shap_abs_mean[:min_len]

        # Global bar plot using Pandas Series (avoids index confusion)
        order = np.argsort(aligned_shap)[::-1]
        shap_series = pd.Series(aligned_shap, index=aligned_features).iloc[order]

        plt.figure(figsize=(9, 5))
        plt.bar(shap_series.index, shap_series.values, color="#4c72b0")
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.title("Global Feature Importance (mean |SHAP|) – Surrogate RF (Cluster Assignment)")
        plt.tight_layout()
        plt.savefig("./explanations/kmeans_digits/shap_global_bar.png", dpi=160)
        plt.close()

        # Beeswarm plot (must use same aligned features)
        # Align X_test columns to match shap_values_for_beeswarm's feature axis
        # If shap_values_for_beeswarm is a list (multi-class), its shape is (n_samples, n_features)
        if isinstance(shap_values_for_beeswarm, list):
            # Defensive: pick the first class again (already chosen above)
            values_for_beeswarm = shap_values_for_beeswarm
        else:
            values_for_beeswarm = shap_values_for_beeswarm

        # Slice X_test to aligned features
        X_test_aligned = X_test[aligned_features]

        shap.summary_plot(values_for_beeswarm, X_test_aligned, feature_names=aligned_features, show=False)
        plt.tight_layout()
        plt.savefig("./explanations/kmeans_digits/shap_beeswarm.png", dpi=160)
        plt.close()

        print("SHAP global plots saved in ./explanations/kmeans_digits/")

        # Single prediction force plot (use aligned features)
        new_idx = X_test_aligned.index[0]
        new_data = X_test_aligned.loc[[new_idx]]
        pred_cluster = kmeans.predict(new_data)[0]

        single_shap_values = explainer.shap_values(new_data, check_additivity=False)
        if isinstance(single_shap_values, list):
            rf_pred = rf.predict(new_data)[0]
            sv = single_shap_values[rf_pred][0]           # shape: (n_features,)
            base = explainer.expected_value[rf_pred]      # scalar base value for that class
        else:
            sv = single_shap_values[0]                    # shape: (n_features,)
            base = explainer.expected_value               # scalar

        sv = np.asarray(sv, dtype=float).ravel()                 # 1D float array
        feat_values = new_data.iloc[0].to_numpy(dtype=float).ravel()

        L = len(aligned_features)
        sv = sv[:L]
        feat_values = feat_values[:L]
        base = float(np.asarray(base).reshape(-1)[0])

        # Debug shapes (optional)
        print(f"[DEBUG] base={base:.6f}, sv.shape={sv.shape}, feat_values.shape={feat_values.shape}, features={L}")

        # Try v0.20+ API first; fallback to legacy if needed; otherwise waterfall PNG
        force_html_path = "./explanations/kmeans_digits/shap_force_single.html"

        # Generate interactive force plot HTML      
        shap_html = shap.plots.force(
            base,                # expected value for the predicted class
            sv,                  # SHAP values for the single point
            features=feat_values,  # actual feature values
            feature_names=aligned_features
        )

        shap.save_html("./explanations/kmeans_digits/shap_force_single.html", shap_html)
        print("Interactive SHAP force plot (single test point) saved as HTML.")

        # Textual explanation for top 3 contributing features
        contributions = list(zip(aligned_features, sv))
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]

        print("=" * 50)
        print(f"Test point predicted KMeans cluster: {pred_cluster}. Top contributing features (surrogate RF):")
        for feat, val in contributions_sorted:
            direction = "positive" if val > 0 else "negative"
            print(f"- {feat} ({direction} impact: {val:.4f})")
        print("=" * 50)


    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


"""
########Output#########

Digits dataset loaded. Rows=1797, Shape=(1797, 65)
Training size: 539, Test size: 1258
Silhouette: 0.1361, Calinski-Harabasz: 79.59, Davies-Bouldin: 1.7822

==================================================
Test point predicted KMeans cluster: 1. Top contributing features (surrogate RF):
- pixel_27 (positive impact: 0.0163)
- pixel_21 (positive impact: 0.0152)
- pixel_26 (negative impact: -0.0131)
==================================================
"""