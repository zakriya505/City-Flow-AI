# src/ml/congestion_classifier.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

CLASS_NAMES = ["Free Flow", "Moderate", "Heavy", "Gridlock"]

def train_congestion_classifier(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model: Gradient Boosted Trees (production choice from plan)
    gbt = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100, # Reduced for faster training in dev
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ])

    print("Training GBT Congestion Classifier...")
    gbt.fit(X_train, y_train)

    y_pred = gbt.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(gbt, "models/congestion_classifier.pkl")
    print("✅ Model saved: models/congestion_classifier.pkl")
    return gbt
