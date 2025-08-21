import numpy as np
from anomalyguard.model import AnomalyGuardModel
from anomalyguard.data import generate_synthetic

def test_model_fit_predict():
    X = generate_synthetic()
    model = AnomalyGuardModel()
    model.fit(X)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({-1, 1})
