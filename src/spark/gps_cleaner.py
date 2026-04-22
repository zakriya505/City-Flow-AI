# src/spark/gps_cleaner.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def clean_taxi_trajectories(spark, input_path: str, output_path: str):
    """
    Clean raw NYC taxi data:
    - Filter invalid coordinates (NYC bounding box)
    - Remove impossible speeds (>120 mph)
    - Handle null values
    - Remove negative fares/distances
    - Deduplicate trip IDs
    """
    print(f"Reading data from {input_path}...")
    try:
        df = spark.read.parquet(input_path)
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return None

    # Step 1: Remove records with null critical columns
    df_clean = df.dropna(subset=[
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID", "trip_distance"
    ])

    # Step 2: Filter valid NYC taxi zone IDs (1-263)
    df_clean = df_clean.filter(
        (F.col("PULocationID").between(1, 263)) &
        (F.col("DOLocationID").between(1, 263))
    )

    # Step 3: Remove physically impossible trips
    df_clean = df_clean \
        .filter(F.col("trip_distance") > 0.1) \
        .filter(F.col("trip_distance") < 100.0) \
        .filter(F.col("fare_amount") > 2.50) \
        .filter(F.col("fare_amount") < 500.0) \
        .filter(F.col("passenger_count").between(1, 6))

    # Step 4: Compute trip duration in minutes
    df_clean = df_clean.withColumn(
        "trip_duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime") -
         F.unix_timestamp("tpep_pickup_datetime")) / 60.0
    )

    # Step 5: Filter valid durations (1 min to 3 hours)
    df_clean = df_clean.filter(
        F.col("trip_duration_min").between(1.0, 180.0)
    )

    # Step 6: Compute average speed (mph)
    df_clean = df_clean.withColumn(
        "avg_speed_mph",
        F.col("trip_distance") / (F.col("trip_duration_min") / 60.0)
    )

    # Step 7: Remove impossible speeds
    df_clean = df_clean.filter(F.col("avg_speed_mph") < 80.0)

    # Step 8: Add time features
    df_clean = df_clean \
        .withColumn("hour_of_day", F.hour("tpep_pickup_datetime")) \
        .withColumn("day_of_week", F.dayofweek("tpep_pickup_datetime")) \
        .withColumn("month", F.month("tpep_pickup_datetime")) \
        .withColumn("is_weekend", F.when(
            F.col("day_of_week").isin([1, 7]), True).otherwise(False)
        )

    # Step 9: Deduplicate
    df_clean = df_clean.dropDuplicates(
        ["tpep_pickup_datetime", "PULocationID", "trip_distance"]
    )

    # Write cleaned data partitioned by month
    print(f"Writing cleaned data to {output_path}...")
    df_clean.write \
        .mode("overwrite") \
        .partitionBy("month") \
        .parquet(output_path)

    print(f"✅ Original records: {df.count():,}")
    print(f"✅ Cleaned records:  {df_clean.count():,}")
    print(f"✅ Retention rate:   {df_clean.count()/df.count()*100:.1f}%")
    return df_clean
