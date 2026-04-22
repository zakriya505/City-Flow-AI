# tests/test_ml_models.py
import pytest
import pandas as pd
import numpy as np
import joblib
import os

def test_congestion_model_loads():
    model_path = "models/congestion_classifier.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
    else:
        pytest.skip("Model file not found")

def test_delay_model_loads():
    model_path = "models/delay_regressor.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
    else:
        pytest.skip("Model file not found")

def test_hotspot_model_loads():
    model_path = "models/hotspot_detector.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
    else:
        pytest.skip("Model file not found")
