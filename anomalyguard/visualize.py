import matplotlib.pyplot as plt

def plot_anomalies(X, y_pred):
    """Scatter plot showing anomalies (-1) vs normal (1)."""
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y_pred, cmap="coolwarm", marker="o", edgecolors="k"
    )
    plt.title("Anomaly Detection Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
