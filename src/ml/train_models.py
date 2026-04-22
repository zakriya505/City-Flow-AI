# src/ml/train_models.py
import os
import sys
import pandas as pd
try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

# Add current dir to path
sys.path.append(os.path.dirname(__file__))

from feature_engineering import build_feature_matrix
from congestion_classifier import train_congestion_classifier
from delay_regressor import train_delay_regressor
from hotspot_detector import train_hotspot_detector

def main():
    if SparkSession is None:
        print("PySpark not installed. Please install it to load aggregated data.")
        return

    spark = SparkSession.builder.appName("CityFlow-Training").getOrCreate()

    AGG_PATH = "data/lake/zone_aggregates/"
    if not os.path.exists(AGG_PATH):
        print(f"Aggregated data not found at {AGG_PATH}. Run batch pipeline first.")
        # Attempt to create dummy data if path doesn't exist for demonstration
        return

    # Load aggregated zone data
    print(f"Loading data from {AGG_PATH}...")
    df_spark = spark.read.parquet(AGG_PATH)
    df = df_spark.toPandas()
    spark.stop()

    print(f"Dataset size: {len(df):,} records across {df['zone_id'].nunique()} zones")

    # Build feature matrix
    df_features = build_feature_matrix(df)

    # All columns except targets and identifiers are features
    TARGET_COLS = ["congestion_level", "delay_minutes", "zone_id", "hour_bucket", "timestamp"]
    FEATURE_COLS = [c for c in df_features.columns if c not in TARGET_COLS]
    X = df_features[FEATURE_COLS]

    print("\n" + "="*60)
    print("TASK 1: Congestion Classification")
    print("="*60)
    train_congestion_classifier(X, df_features["congestion_level"])

    print("\n" + "="*60)
    print("TASK 2: Travel Delay Regression")
    print("="*60)
    train_delay_regressor(X, df_features["delay_minutes"])

    print("\n" + "="*60)
    print("TASK 3: Hotspot Detection")
    print("="*60)
    train_hotspot_detector(X)

    print("\n✅ All models trained and saved to models/")

if __name__ == "__main__":
    main()
