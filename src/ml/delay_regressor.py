# src/ml/delay_regressor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_delay_regressor(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gbr = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            min_samples_leaf=10,
            random_state=42
        ))
    ])
    
    print("Training GBT Delay Regressor...")
    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"  RMSE: {rmse:.3f} minutes")
    print(f"  MAE:  {mae:.3f} minutes")
    print(f"  R²:   {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(gbr, "models/delay_regressor.pkl")
    print("✅ Model saved: models/delay_regressor.pkl")
    return gbr
