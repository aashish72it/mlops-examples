
import os
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Stage 1: Load UCI Wholesale Customers Dataset ----------------
def load_uci_wholesale_customers():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    df = pd.read_csv(url)
    print("[INFO] Loaded UCI Wholesale Customers dataset.")
    return df

if __name__ == "__main__":
    try:
        # Load dataset
        df = load_uci_wholesale_customers()
        print(f"Dataset shape: {df.shape}")
        print("=" * 80)
        print(df.head())
        print("=" * 80)

        # ---------------- Stage 2: Feature Selection & Train-Test Split ----------------
        feature_names = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        X = df[feature_names]

        stratify_labels = df['Channel'] if 'Channel' in df.columns else None

        if stratify_labels is not None:
            X_train_raw, X_test_raw = train_test_split(X, test_size=0.7, random_state=42, stratify=stratify_labels)
        else:
            X_train_raw, X_test_raw = train_test_split(X, test_size=0.7, random_state=42)

        print(f"Training size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

        # Scale features
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names)
        X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names)

        # ---------------- Stage 3: Train KMeans Model ----------------
        kmeans = KMeans(n_clusters=5, n_init="auto", random_state=42)
        kmeans.fit(X_train)

        train_clusters = kmeans.predict(X_train)
        test_clusters = kmeans.predict(X_test)

        # ---------------- Stage 4: Evaluate Clustering ----------------
        sil = silhouette_score(X_test, test_clusters)
        ch = calinski_harabasz_score(X_test, test_clusters)
        db = davies_bouldin_score(X_test, test_clusters)

        print(f"Silhouette: {sil:.4f}, Calinski-Harabasz: {ch:.2f}, Davies-Bouldin: {db:.4f}")
        print("Cluster sizes (test set):", dict(zip(*np.unique(test_clusters, return_counts=True))))

        # ---------------- Stage 5: Save Model ----------------
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        with open("./trained_models/wholesale_kmeans.pkl", "wb") as f:
            pickle.dump({"kmeans": kmeans, "scaler": scaler, "features": feature_names}, f)

        # ---------------- Stage 6: Visualize Clusters ----------------
        Path("./explanations/wholesale_kmeans/").mkdir(parents=True, exist_ok=True)
        pca = PCA(n_components=2, random_state=42)
        X_test_pca = pca.fit_transform(X_test)
        df_plot = pd.DataFrame(X_test_pca, columns=["PC1", "PC2"], index=X_test.index)
        df_plot["cluster"] = test_clusters

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="cluster", palette="Set2", s=60)
        plt.title("Wholesale Customers Clusters (PCA 2D)")
        plt.savefig("./explanations/wholesale_kmeans/wholesale_clusters.png", dpi=160)
        plt.close()

        # ---------------- Stage 7: SHAP Explanations ----------------
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train[feature_names], train_clusters)

        background = X_train[feature_names].sample(n=min(100, len(X_train)), random_state=42)
        explainer = shap.TreeExplainer(rf, data=background)
        shap_values = explainer.shap_values(X_test[feature_names], check_additivity=False)

        # Handle multi-class SHAP outputs
        if isinstance(shap_values, list):
            shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            shap_values_for_beeswarm = shap_values[0]
        else:
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
            shap_values_for_beeswarm = shap_values

        shap_abs_mean = np.asarray(shap_abs_mean).flatten()
        aligned_features = feature_names[:len(shap_abs_mean)]

        # Global bar plot
        order = np.argsort(shap_abs_mean)[::-1]
        plt.figure(figsize=(7.5, 5))
        plt.bar([aligned_features[i] for i in order], shap_abs_mean[order], color="#4c72b0")
        plt.xticks(rotation=30, ha="right")
        plt.title("Global Feature Importance (mean |SHAP|) - Surrogate RF")
        plt.tight_layout()
        plt.savefig("./explanations/wholesale_kmeans/shap_global_bar.png", dpi=160)
        plt.close()

        # Beeswarm plot
        shap.summary_plot(shap_values_for_beeswarm, X_test[aligned_features], feature_names=aligned_features, show=False)
        plt.tight_layout()
        plt.savefig("./explanations/wholesale_kmeans/shap_beeswarm.png", dpi=160)
        plt.close()

        print("SHAP explanations saved in ./explanations/wholesale_kmeans/")

        # Single prediction force plot
        new_data = X_test.iloc[[0]]
        rf_pred = rf.predict(new_data)[0]
        single_shap_values = explainer.shap_values(new_data)
        sv = single_shap_values[rf_pred][0]
        base = explainer.expected_value[rf_pred]

        try:
            shap_html = shap.plots.force(base, sv, features=new_data.iloc[0].values, feature_names=aligned_features)
            shap.save_html("./explanations/wholesale_kmeans/shap_force_single.html", shap_html)
            print("Interactive SHAP force plot saved as HTML.")
        except Exception as e:
            print(f"[WARN] Force plot failed: {e}. Falling back to waterfall plot.")
            exp = shap.Explanation(values=sv, base_values=base, data=new_data.iloc[0].values, feature_names=aligned_features)
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(exp, show=False)
            plt.tight_layout()
            plt.savefig("./explanations/wholesale_kmeans/shap_waterfall_single.png", dpi=160)
            plt.close()
            print("Saved SHAP waterfall plot as fallback.")

        # Textual explanation for top 3 contributing features
        contributions = list(zip(aligned_features, sv))
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]

        print("=" * 50)
        print(f"Predicted cluster: {rf_pred}. Top contributing features:")
        for feat, val in contributions_sorted:
            direction = "positive" if val > 0 else "negative"
            print(f"- {feat} ({direction} impact: {val:.4f})")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
