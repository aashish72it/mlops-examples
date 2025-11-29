
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

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
