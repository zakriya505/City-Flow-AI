# src/spark/batch_processing.py
from pyspark.sql import SparkSession
import os
import sys

# Add current dir to path to import local modules
sys.path.append(os.path.dirname(__file__))

from gps_cleaner import clean_taxi_trajectories
from zone_mapper import aggregate_by_zone_hour

def run_batch_pipeline():
    spark = SparkSession.builder \
        .appName("CityFlow-BatchPipeline") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.parquet.filterPushdown", "true") \
        .config("spark.sql.parquet.mergeSchema", "false") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    RAW_PATH = "data/raw/taxi_rides/"
    CLEAN_PATH = "data/lake/trajectory_cleaned/"
    AGG_PATH = "data/lake/zone_aggregates/"

    # Stage 1: Clean
    df_clean = clean_taxi_trajectories(spark, RAW_PATH, CLEAN_PATH)
    if df_clean is None:
        print("Cleaning failed. Exiting.")
        return

    # Stage 2: Aggregate
    df_agg = aggregate_by_zone_hour(df_clean, AGG_PATH)

    # Stage 3: Write to PostgreSQL
    db_url = os.getenv("POSTGRES_URI", "jdbc:postgresql://localhost:5432/cityflow")
    db_props = {"user": "cityflow_user", "password": "cityflow_pass",
                "driver": "org.postgresql.Driver"}

    print("Writing aggregated results to PostgreSQL...")
    try:
        # Select only columns present in DB table
        db_cols = ["zone_id", "timestamp", "avg_speed", "trip_count",
                   "congestion_level", "hour_of_day", "day_of_week"]
        df_agg.select(*db_cols).write \
            .mode("append") \
            .jdbc(db_url, "zone_metrics", properties=db_props)
        print("✅ Batch pipeline complete.")
    except Exception as e:
        print(f"Failed to write to PostgreSQL: {e}")
        print("Note: PostgreSQL might not be running or driver might be missing.")

    spark.stop()

if __name__ == "__main__":
    run_batch_pipeline()
