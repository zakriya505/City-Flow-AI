# src/ml/hotspot_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def train_hotspot_detector(X: pd.DataFrame):
    """
    Unsupervised anomaly detection to find emerging traffic hotspots.
    Hotspots = zones with sudden spike in trip density + speed drop.
    """
    # Key features for anomaly detection
    HOTSPOT_FEATURES = [
        "trip_count", "avg_speed", "speed_delta",
        "trips_per_minute", "congestion_ratio",
        "speed_to_zone_avg", "lag1_speed"
    ]

    # Use only subset of features that exist in X
    features_to_use = [f for f in HOTSPOT_FEATURES if f in X.columns]
    X_hot = X[features_to_use]

    iso_forest = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=100,
            contamination=0.05,      # Expect ~5% hotspot events
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training Isolation Forest Hotspot Detector...")
    iso_forest.fit(X_hot)

    predictions = iso_forest.predict(X_hot)
    hotspot_flags = (predictions == -1).astype(int)

    hotspot_pct = hotspot_flags.mean() * 100
    print(f"Hotspot detection rate: {hotspot_pct:.2f}%")
    print(f"Total hotspots detected: {hotspot_flags.sum():,}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(iso_forest, "models/hotspot_detector.pkl")
    print("✅ Model saved: models/hotspot_detector.pkl")
    return iso_forest
