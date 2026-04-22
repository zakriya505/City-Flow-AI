# src/spark/zone_mapper.py
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def aggregate_by_zone_hour(df_clean, output_path: str):
    """
    Aggregate trip-level data into zone × hour grid.
    This creates the base features for ML models.
    """
    print("Aggregating data by zone and hour...")
    zone_agg = df_clean.groupBy(
        F.col("PULocationID").alias("zone_id"),
        F.date_trunc("hour", "tpep_pickup_datetime").alias("hour_bucket")
    ).agg(
        F.count("*").alias("trip_count"),
        F.avg("avg_speed_mph").alias("avg_speed"),
        F.stddev("avg_speed_mph").alias("speed_stddev"),
        F.avg("trip_distance").alias("avg_distance"),
        F.avg("trip_duration_min").alias("avg_duration"),
        F.avg("fare_amount").alias("avg_fare"),
        F.sum("passenger_count").alias("total_passengers"),
        F.first("hour_of_day").alias("hour_of_day"),
        F.first("day_of_week").alias("day_of_week"),
        F.first("is_weekend").alias("is_weekend")
    )

    # Compute congestion label based on speed thresholds
    # NYC DOT definitions: Free=25+mph, Moderate=15-25mph, Heavy=8-15mph, Gridlock=<8mph
    zone_agg = zone_agg.withColumn(
        "congestion_level",
        F.when(F.col("avg_speed") >= 25.0, 0)     # Free flow
         .when(F.col("avg_speed") >= 15.0, 1)     # Moderate
         .when(F.col("avg_speed") >= 8.0, 2)      # Heavy
         .otherwise(3)                             # Gridlock
    )

    # Lag features (previous hour's metrics for same zone)
    zone_window = Window.partitionBy("zone_id") \
                        .orderBy("hour_bucket")

    zone_agg = zone_agg \
        .withColumn("prev_hour_speed", F.lag("avg_speed", 1).over(zone_window)) \
        .withColumn("prev_hour_trips", F.lag("trip_count", 1).over(zone_window)) \
        .withColumn("speed_delta", F.col("avg_speed") - F.col("prev_hour_speed"))

    # Add timestamp column for DB
    zone_agg = zone_agg.withColumn("timestamp", F.col("hour_bucket"))

    print(f"Writing aggregated data to {output_path}...")
    zone_agg.write \
        .mode("overwrite") \
        .partitionBy("hour_of_day") \
        .parquet(output_path)

    return zone_agg
