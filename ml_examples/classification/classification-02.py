
# classification-01.py (top of file)
import os
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


import argparse
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def dataset_download(url: str, dest: str):
    """Download a file using built-in urllib (no external tools)."""
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("Done.")


def random_sample(df: pd.DataFrame, frac: float = 0.2, random_state: int = 42):
    """Simple random sampling by fraction."""
    return df.sample(frac=frac, random_state=random_state)


def load_iris(path="./datasets/iris.csv"):
    cols = ["sepal_length","sepal_width","petal_length","petal_width","class"]
    df = pd.read_csv(path, header=None, names=cols)
    df = df.dropna()
    X = df[cols[:-1]]
    y = df["class"]
    return X, y, df



if __name__ == "__main__":
    dataset_download("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
             "./datasets/wine.csv")
