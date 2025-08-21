import pandas as pd
from sklearn.datasets import make_blobs

def load_csv(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)

def generate_synthetic(n_samples=300, n_features=2, n_clusters=2, random_state=42):
    """Generate synthetic clustering dataset."""
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return X
