from anomalyguard import AnomalyDetector, generate_synthetic

def test_anomaly_detector():
    data, labels = generate_synthetic()
    detector = AnomalyDetector()
    detector.fit(data)

    preds = detector.predict(data)
    assert len(preds) == len(labels)
