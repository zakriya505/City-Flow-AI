# src/spark/streaming_pipeline.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

def run_streaming_pipeline():
    spark = SparkSession.builder \
        .appName("CityFlow-Streaming") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/cityflow_checkpoints") \
        .getOrCreate()

    # Define schema for Kafka taxi trip messages
    TRIP_SCHEMA = StructType([
        StructField("trip_id", StringType()),
        StructField("pickup_zone", IntegerType()),
        StructField("dropoff_zone", IntegerType()),
        StructField("pickup_datetime", StringType()),
        StructField("passenger_count", IntegerType()),
        StructField("trip_distance", DoubleType()),
        StructField("fare_amount", DoubleType())
    ])

    # Read from Kafka
    try:
        raw_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "city.taxi.trips,city.traffic.sensors") \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", 10000) \
            .load()
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        return

    # Parse JSON payload
    trip_stream = raw_stream \
        .filter(F.col("topic") == "city.taxi.trips") \
        .select(
            F.from_json(
                F.col("value").cast("string"),
                TRIP_SCHEMA
            ).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        ).select("data.*", "kafka_timestamp")

    # Add event time watermark (handle late arrivals up to 2 min)
    trip_stream = trip_stream \
        .withColumn("event_time",
                    F.to_timestamp("pickup_datetime")) \
        .withWatermark("event_time", "2 minutes")

    # 5-minute window aggregations per zone
    window_agg = trip_stream \
        .groupBy(
            F.col("pickup_zone").alias("zone_id"),
            F.window("event_time", "5 minutes", "1 minute")
        ).agg(
            F.count("*").alias("trip_count_5min"),
            F.avg("trip_distance").alias("avg_distance_5min"),
            F.sum("passenger_count").alias("total_passengers_5min")
        ) \
        .select(
            "zone_id",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "trip_count_5min",
            "avg_distance_5min",
            "total_passengers_5min"
        )

    # Write aggregated windows to PostgreSQL
    def write_to_postgres(batch_df, batch_id):
        if batch_df.count() > 0:
            print(f"Streaming batch {batch_id}: writing {batch_df.count()} records.")
            try:
                db_url = "jdbc:postgresql://localhost:5432/cityflow"
                batch_df.write.mode("append").jdbc(
                    db_url, "stream_zone_windows",
                    properties={"user": "cityflow_user",
                                "password": "cityflow_pass",
                                "driver": "org.postgresql.Driver"}
                )
            except Exception as e:
                print(f"Streaming DB write failed: {e}")

    query = window_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(write_to_postgres) \
        .trigger(processingTime="30 seconds") \
        .start()

    print("🚀 Streaming pipeline started. Waiting for data...")
    query.awaitTermination()

if __name__ == "__main__":
    run_streaming_pipeline()
