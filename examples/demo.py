from anomalyguard.data import generate_synthetic
from anomalyguard.model import AnomalyGuardModel
from anomalyguard.visualize import plot_anomalies

# 1. Load synthetic dataset
X = generate_synthetic()

# 2. Train model
model = AnomalyGuardModel(contamination=0.1)
model.fit(X)

# 3. Predictions
y_pred = model.predict(X)

# 4. Plot anomalies
plot_anomalies(X, y_pred)
