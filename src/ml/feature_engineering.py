# src/ml/feature_engineering.py
import pandas as pd
import numpy as np

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive feature set for all three ML tasks.
    Input: zone_aggregates DataFrame
    Output: feature matrix ready for training
    """
    df = df.copy()

    # Ensure sorting for lag features
    df = df.sort_values(["zone_id", "hour_bucket"])

    # --- Temporal Features ---
    df["is_rush_hour"] = df["hour_of_day"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )
    df["is_overnight"] = df["hour_of_day"].apply(
        lambda h: 1 if h < 6 or h >= 22 else 0
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Speed Ratio Features ---
    df["speed_to_zone_avg"] = df.groupby("zone_id")["avg_speed"] \
                                 .transform(lambda x: x / (x.mean() + 1e-6))

    # --- Lag Features ---
    df["lag1_speed"] = df.groupby("zone_id")["avg_speed"].shift(1)
    df["lag2_speed"] = df.groupby("zone_id")["avg_speed"].shift(2)
    df["lag1_trips"] = df.groupby("zone_id")["trip_count"].shift(1)
    df["rolling_avg_speed_3h"] = df.groupby("zone_id")["avg_speed"] \
                                    .transform(lambda x: x.rolling(3, min_periods=1).mean())

    # --- Traffic Density Features ---
    df["trips_per_minute"] = df["trip_count"] / 60.0
    df["congestion_ratio"] = df["trip_count"] / (df["avg_speed"] + 1e-6)

    # --- Target Variables ---
    # Task 2: Delay in minutes (avg_duration - theoretical min duration)
    # Assume 30mph is free flow speed
    df["theoretical_min_duration"] = df["avg_distance"] / 30.0 * 60
    df["delay_minutes"] = np.maximum(
        0, df["avg_duration"] - df["theoretical_min_duration"]
    )

    # Fill NaNs from lag features
    df = df.fillna(method="bfill").fillna(0)

    FEATURE_COLS = [
        "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour",
        "is_overnight", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "trip_count", "avg_speed", "speed_stddev", "avg_distance",
        "speed_delta", "lag1_speed", "lag2_speed", "lag1_trips",
        "rolling_avg_speed_3h", "speed_to_zone_avg",
        "trips_per_minute", "congestion_ratio", "total_passengers"
    ]

    # Re-check columns exist
    existing_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    return df[existing_cols + ["congestion_level", "delay_minutes", "zone_id"]]
