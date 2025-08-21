from sklearn.ensemble import IsolationForest

class AnomalyGuardModel:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X):
        """Train the anomaly detector."""
        self.model.fit(X)

    def predict(self, X):
        """Predict anomalies (-1 = anomaly, 1 = normal)."""
        return self.model.predict(X)

    def anomaly_scores(self, X):
        """Return anomaly scores (lower = more abnormal)."""
        return self.model.decision_function(X)
