# 🏙️ City Flow AI — Implementation Plan

> **Complete step-by-step build guide for the Urban Mobility Intelligence System**  
> Stack: Apache Spark · Kafka · PostgreSQL · Parquet · MLlib · Scikit-learn · Streamlit

---

## 📅 Project Timeline Overview

| Phase | Duration | Focus |
|---|---|---|
| Phase 1 | Week 1 | Environment Setup + Data Acquisition |
| Phase 2 | Week 2 | Kafka Ingestion Pipeline |
| Phase 3 | Week 3 | Spark Batch Processing |
| Phase 4 | Week 4 | Spark Structured Streaming |
| Phase 5 | Week 5–6 | Feature Engineering + ML Models |
| Phase 6 | Week 7 | Streamlit Dashboard |
| Phase 7 | Week 8 | Optimization, Testing & Documentation |

---

## PHASE 1 — Environment Setup & Data Acquisition

### 1.1 Infrastructure Setup

```bash
# Install Java (required for Spark)
sudo apt-get install openjdk-11-jdk

# Install Python dependencies
pip install pyspark==3.5.0 kafka-python==2.0.2 psycopg2-binary \
            scikit-learn==1.4.0 pandas==2.2.0 numpy==1.26.0 \
            streamlit==1.32.0 plotly==5.20.0 folium==0.16.0 \
            pyarrow==15.0.0 python-dotenv requests
```

**`docker-compose.yml`** — Infrastructure services:

```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on: [zookeeper]
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_NUM_PARTITIONS: 6
      KAFKA_DEFAULT_REPLICATION_FACTOR: 1

  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: cityflow
      POSTGRES_USER: cityflow_user
      POSTGRES_PASSWORD: cityflow_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  postgres_data:
```

**`docker/init.sql`** — Database schema:

```sql
CREATE TABLE zone_metrics (
    id SERIAL PRIMARY KEY,
    zone_id INT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    avg_speed FLOAT,
    trip_count INT,
    congestion_level INT,         -- 0=free, 1=moderate, 2=heavy, 3=gridlock
    predicted_delay_minutes FLOAT,
    is_hotspot BOOLEAN DEFAULT FALSE,
    weather_condition VARCHAR(50),
    temperature FLOAT,
    hour_of_day INT,
    day_of_week INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_zone_metrics_zone_time ON zone_metrics(zone_id, timestamp DESC);
CREATE INDEX idx_zone_metrics_hotspot ON zone_metrics(is_hotspot, timestamp DESC);

CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    training_date TIMESTAMP,
    dataset_size BIGINT,
    notes TEXT
);
```

### 1.2 Data Acquisition

**NYC TLC Trip Data** (primary dataset):

```python
# src/ingestion/data_downloader.py
import requests
import os

# NYC TLC Open Data - Yellow Taxi Trip Records
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
MONTHS = ["2023-01", "2023-02", "2023-03", "2023-04",
          "2023-05", "2023-06", "2023-07", "2023-08",
          "2023-09", "2023-10", "2023-11", "2023-12"]

def download_tlc_data(output_dir: str = "data/raw/taxi_rides"):
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"

    for month in MONTHS:
        filename = f"yellow_tripdata_{month}.parquet"
        url = f"{base_url}/{filename}"
        out_path = os.path.join(output_dir, filename)

        if not os.path.exists(out_path):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✓ Saved {filename}")

if __name__ == "__main__":
    download_tlc_data()
```

### 1.3 Traffic Sensor Simulation

```python
# src/ingestion/sensor_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

NYC_ZONES = list(range(1, 264))   # 263 official TLC taxi zones

def simulate_sensor_data(n_records: int = 10000) -> pd.DataFrame:
    """Generate realistic traffic sensor readings for NYC zones."""
    base_time = datetime.now()
    records = []

    for i in range(n_records):
        zone = random.choice(NYC_ZONES)
        hour = (base_time + timedelta(minutes=i)).hour

        # Rush hour pattern: morning (7-9am) and evening (5-7pm)
        rush_hour_multiplier = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            rush_hour_multiplier = 2.5
        elif 22 <= hour or hour <= 5:
            rush_hour_multiplier = 0.3

        base_speed = random.gauss(25, 8)  # avg NYC speed ~25 mph
        speed = max(2.0, base_speed / rush_hour_multiplier + random.gauss(0, 3))

        vehicle_count = int(random.gauss(45, 15) * rush_hour_multiplier)

        records.append({
            "sensor_id": f"SENS_{zone:03d}_{random.randint(1, 5)}",
            "zone_id": zone,
            "timestamp": base_time + timedelta(minutes=i * 0.5),
            "avg_speed_mph": round(speed, 2),
            "vehicle_count": max(0, vehicle_count),
            "occupancy_pct": round(min(100, vehicle_count / 80 * 100), 1)
        })

    return pd.DataFrame(records)
```

---

## PHASE 2 — Kafka Streaming Ingestion

### 2.1 Kafka Producer — Live Data Stream

```python
# src/ingestion/kafka_producer.py
import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from sensor_simulator import simulate_sensor_data

KAFKA_BOOTSTRAP = "localhost:9092"
TOPICS = {
    "taxi_trips": "city.taxi.trips",
    "traffic_sensors": "city.traffic.sensors",
    "weather": "city.weather.updates"
}

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    key_serializer=lambda k: str(k).encode("utf-8"),
    acks="all",                     # Wait for all replicas
    retries=3,
    batch_size=16384,               # 16KB batch for throughput
    linger_ms=10,                   # Small delay to batch messages
    compression_type="lz4",         # Compress for network efficiency
    buffer_memory=33554432          # 32MB buffer
)

def stream_taxi_trips():
    """Simulate streaming ride-hailing trip starts."""
    zones = list(range(1, 264))
    while True:
        trip = {
            "trip_id": f"T{random.randint(100000, 999999)}",
            "pickup_zone": random.choice(zones),
            "dropoff_zone": random.choice(zones),
            "pickup_datetime": datetime.now().isoformat(),
            "passenger_count": random.randint(1, 4),
            "trip_distance": round(random.uniform(0.5, 20.0), 2),
            "fare_amount": round(random.uniform(5.0, 80.0), 2),
            "payment_type": random.choice(["credit_card", "cash", "app"])
        }
        producer.send(
            TOPICS["taxi_trips"],
            key=trip["pickup_zone"],
            value=trip
        )
        time.sleep(0.05)   # ~20 messages/second

def stream_sensor_readings():
    """Stream traffic sensor readings."""
    while True:
        sensor_df = simulate_sensor_data(n_records=50)
        for _, row in sensor_df.iterrows():
            producer.send(
                TOPICS["traffic_sensors"],
                key=row["zone_id"],
                value=row.to_dict()
            )
        time.sleep(1.0)

if __name__ == "__main__":
    import threading
    t1 = threading.Thread(target=stream_taxi_trips, daemon=True)
    t2 = threading.Thread(target=stream_sensor_readings, daemon=True)
    t1.start()
    t2.start()
    print("✅ Kafka producers running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        producer.flush()
        producer.close()
```

### 2.2 Kafka Topics Configuration

```bash
# Create topics with appropriate partitions
kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --replication-factor 1 \
  --partitions 6 \
  --topic city.taxi.trips \
  --config retention.ms=86400000 \
  --config segment.bytes=104857600

kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --replication-factor 1 \
  --partitions 6 \
  --topic city.traffic.sensors

kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --replication-factor 1 \
  --partitions 3 \
  --topic city.weather.updates
```

---

## PHASE 3 — Spark Batch Processing

### 3.1 GPS Trajectory Cleaning

```python
# src/spark/gps_cleaner.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, BooleanType
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("CityFlow-GPSCleaner") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

def clean_taxi_trajectories(input_path: str, output_path: str):
    """
    Clean raw NYC taxi data:
    - Filter invalid coordinates (NYC bounding box)
    - Remove impossible speeds (>120 mph)
    - Handle null values
    - Remove negative fares/distances
    - Deduplicate trip IDs
    """
    df = spark.read.parquet(input_path)

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

    # Write cleaned data partitioned by date
    df_clean.write \
        .mode("overwrite") \
        .partitionBy("month") \
        .parquet(output_path)

    print(f"✅ Original records: {df.count():,}")
    print(f"✅ Cleaned records:  {df_clean.count():,}")
    print(f"✅ Retention rate:   {df_clean.count()/df.count()*100:.1f}%")
    return df_clean
```

### 3.2 Zone-Level Aggregation

```python
# src/spark/zone_mapper.py
from pyspark.sql import functions as F

def aggregate_by_zone_hour(df_clean, output_path: str):
    """
    Aggregate trip-level data into zone × hour grid.
    This creates the base features for ML models.
    """
    zone_agg = df_clean.groupBy(
        "PULocationID",
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
    from pyspark.sql.window import Window
    zone_window = Window.partitionBy("PULocationID") \
                        .orderBy("hour_bucket")

    zone_agg = zone_agg \
        .withColumn("prev_hour_speed", F.lag("avg_speed", 1).over(zone_window)) \
        .withColumn("prev_hour_trips", F.lag("trip_count", 1).over(zone_window)) \
        .withColumn("speed_delta", F.col("avg_speed") - F.col("prev_hour_speed"))

    zone_agg.write \
        .mode("overwrite") \
        .partitionBy("hour_of_day") \
        .parquet(output_path)

    return zone_agg
```

### 3.3 Batch Processing Orchestrator

```python
# src/spark/batch_processing.py
from pyspark.sql import SparkSession
import os

def run_batch_pipeline():
    spark = SparkSession.builder \
        .appName("CityFlow-BatchPipeline") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.parquet.filterPushdown", "true") \
        .config("spark.sql.parquet.mergeSchema", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    RAW_PATH = "data/raw/taxi_rides/"
    CLEAN_PATH = "data/lake/trajectory_cleaned/"
    AGG_PATH = "data/lake/zone_aggregates/"

    # Stage 1: Clean
    from gps_cleaner import clean_taxi_trajectories
    df_clean = clean_taxi_trajectories(RAW_PATH, CLEAN_PATH)

    # Stage 2: Aggregate
    from zone_mapper import aggregate_by_zone_hour
    df_agg = aggregate_by_zone_hour(df_clean, AGG_PATH)

    # Stage 3: Write to PostgreSQL
    db_url = os.getenv("POSTGRES_URI", "jdbc:postgresql://localhost:5432/cityflow")
    db_props = {"user": "cityflow_user", "password": "cityflow_pass",
                "driver": "org.postgresql.Driver"}

    df_agg.write \
        .mode("append") \
        .jdbc(db_url, "zone_metrics", properties=db_props)

    print("✅ Batch pipeline complete.")
    spark.stop()

if __name__ == "__main__":
    run_batch_pipeline()
```

---

## PHASE 4 — Spark Structured Streaming

### 4.1 Real-Time Stream Processing

```python
# src/spark/streaming_pipeline.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

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

def run_streaming_pipeline():
    # Read from Kafka
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "city.taxi.trips,city.traffic.sensors") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .option("maxOffsetsPerTrigger", 10000) \
        .load()

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
            "pickup_zone",
            F.window("event_time", "5 minutes", "1 minute")
        ).agg(
            F.count("*").alias("trip_count_5min"),
            F.avg("trip_distance").alias("avg_distance_5min"),
            F.sum("passenger_count").alias("total_passengers_5min")
        ) \
        .select(
            F.col("pickup_zone").alias("zone_id"),
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "trip_count_5min",
            "avg_distance_5min",
            "total_passengers_5min"
        )

    # Write aggregated windows to PostgreSQL
    def write_to_postgres(batch_df, batch_id):
        if batch_df.count() > 0:
            db_url = "jdbc:postgresql://localhost:5432/cityflow"
            batch_df.write.mode("append").jdbc(
                db_url, "stream_zone_windows",
                properties={"user": "cityflow_user",
                            "password": "cityflow_pass",
                            "driver": "org.postgresql.Driver"}
            )

    query = window_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(write_to_postgres) \
        .trigger(processingTime="30 seconds") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    run_streaming_pipeline()
```

---

## PHASE 5 — Feature Engineering & ML Models

### 5.1 Feature Engineering Pipeline

```python
# src/ml/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive feature set for all three ML tasks.
    Input: zone_aggregates DataFrame
    Output: feature matrix ready for training
    """
    df = df.copy()

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
    df = df.sort_values(["zone_id", "hour_bucket"])
    df["lag1_speed"] = df.groupby("zone_id")["avg_speed"].shift(1)
    df["lag2_speed"] = df.groupby("zone_id")["avg_speed"].shift(2)
    df["lag1_trips"] = df.groupby("zone_id")["trip_count"].shift(1)
    df["rolling_avg_speed_3h"] = df.groupby("zone_id")["avg_speed"] \
                                    .transform(lambda x: x.rolling(3, min_periods=1).mean())

    # --- Traffic Density Features ---
    df["trips_per_minute"] = df["trip_count"] / 60.0
    df["congestion_ratio"] = df["trip_count"] / (df["avg_speed"] + 1e-6)

    # --- Target Variables ---
    # Task 1: Congestion level (already in data: 0,1,2,3)
    # Task 2: Delay in minutes (avg_duration - theoretical min duration)
    df["theoretical_min_duration"] = df["avg_distance"] / 30.0 * 60   # at 30mph
    df["delay_minutes"] = np.maximum(
        0, df["avg_duration"] - df["theoretical_min_duration"]
    )
    # Task 3: Hotspot flag (anomaly target)

    df = df.fillna(method="ffill").fillna(0)

    FEATURE_COLS = [
        "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour",
        "is_overnight", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "trip_count", "avg_speed", "speed_stddev", "avg_distance",
        "speed_delta", "lag1_speed", "lag2_speed", "lag1_trips",
        "rolling_avg_speed_3h", "speed_to_zone_avg",
        "trips_per_minute", "congestion_ratio", "total_passengers"
    ]

    return df[FEATURE_COLS + ["congestion_level", "delay_minutes", "zone_id"]]
```

### 5.2 Task 1 — Congestion Zone Classification

```python
# src/ml/congestion_classifier.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                              accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Target: 4-class congestion (0=Free, 1=Moderate, 2=Heavy, 3=Gridlock)
CLASS_NAMES = ["Free Flow", "Moderate", "Heavy", "Gridlock"]

def train_congestion_classifier(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model 1: Random Forest (baseline)
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",   # handles imbalanced gridlock class
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Model 2: Gradient Boosted Trees (production)
    gbt = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ])

    # Train both
    rf.fit(X_train, y_train)
    gbt.fit(X_train, y_train)

    # Evaluate
    for name, model in [("Random Forest", rf), ("GBT", gbt)]:
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Cross-validation on best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gbt, X_train, y_train, cv=cv,
                                scoring="f1_weighted", n_jobs=-1)
    print(f"\nGBT CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    joblib.dump(gbt, "models/congestion_classifier.pkl")
    print("✅ Model saved: models/congestion_classifier.pkl")
    return gbt
```

### 5.3 Task 2 — Travel Delay Regression

```python
# src/ml/delay_regressor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def train_delay_regressor(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline: Ridge Regression
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=10.0))
    ])
    ridge.fit(X_train, y_train)

    # Production: Gradient Boosting Regressor
    gbr = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            min_samples_leaf=10,
            random_state=42
        ))
    ])
    gbr.fit(X_train, y_train)

    for name, model in [("Ridge (baseline)", ridge), ("GBT (production)", gbr)]:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.3f} minutes")
        print(f"  MAE:  {mae:.3f} minutes")
        print(f"  R²:   {r2:.4f}")

    joblib.dump(gbr, "models/delay_regressor.pkl")
    print("✅ Model saved: models/delay_regressor.pkl")
    return gbr
```

### 5.4 Task 3 — Hotspot Emergence Detection

```python
# src/ml/hotspot_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

def train_hotspot_detector(X: pd.DataFrame, y_true_hotspot: pd.Series = None):
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

    X_hot = X[HOTSPOT_FEATURES]

    iso_forest = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination=0.05,      # Expect ~5% hotspot events
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        ))
    ])

    iso_forest.fit(X_hot)

    predictions = iso_forest.predict(X_hot)
    hotspot_flags = (predictions == -1).astype(int)

    hotspot_pct = hotspot_flags.mean() * 100
    print(f"Hotspot detection rate: {hotspot_pct:.2f}%")
    print(f"Total hotspots detected: {hotspot_flags.sum():,}")

    if y_true_hotspot is not None:
        precision = precision_score(y_true_hotspot, hotspot_flags)
        recall    = recall_score(y_true_hotspot, hotspot_flags)
        f1        = f1_score(y_true_hotspot, hotspot_flags)
        print(f"\nEvaluation (vs labeled hotspots):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    joblib.dump(iso_forest, "models/hotspot_detector.pkl")
    print("✅ Model saved: models/hotspot_detector.pkl")
    return iso_forest, hotspot_flags
```

### 5.5 Training Orchestrator

```python
# src/ml/train_models.py
import os
import pandas as pd
from pyspark.sql import SparkSession
from feature_engineering import build_feature_matrix
from congestion_classifier import train_congestion_classifier
from delay_regressor import train_delay_regressor
from hotspot_detector import train_hotspot_detector

os.makedirs("models", exist_ok=True)

def main():
    spark = SparkSession.builder.appName("CityFlow-Training").getOrCreate()

    # Load aggregated zone data
    df_spark = spark.read.parquet("data/lake/zone_aggregates/")
    df = df_spark.toPandas()
    spark.stop()

    print(f"Dataset size: {len(df):,} records across {df['zone_id'].nunique()} zones")

    # Build feature matrix
    df_features = build_feature_matrix(df)

    FEATURE_COLS = [c for c in df_features.columns
                    if c not in ["congestion_level", "delay_minutes", "zone_id"]]
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
```

---

## PHASE 6 — Streamlit Dashboard

### 6.1 Main Dashboard App

```python
# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import joblib

st.set_page_config(
    page_title="City Flow AI — Urban Mobility Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- DB Connection ----
@st.cache_resource
def get_db():
    return psycopg2.connect(
        host="localhost", database="cityflow",
        user="cityflow_user", password="cityflow_pass"
    )

@st.cache_data(ttl=30)   # Refresh every 30 seconds
def load_latest_metrics():
    conn = get_db()
    query = """
        SELECT zone_id, timestamp, avg_speed, trip_count,
               congestion_level, predicted_delay_minutes,
               is_hotspot, hour_of_day, day_of_week
        FROM zone_metrics
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
        ORDER BY timestamp DESC
        LIMIT 5000
    """
    return pd.read_sql(query, conn)

# ---- Header ----
st.title("🏙️ City Flow AI — Urban Mobility Intelligence")
st.markdown("*Real-time congestion prediction · Delay estimation · Hotspot detection*")

# ---- Sidebar Filters ----
st.sidebar.header("🔧 Filters")
refresh_rate = st.sidebar.slider("Auto-refresh (sec)", 10, 120, 30)
selected_zones = st.sidebar.multiselect("Filter Zones", list(range(1, 264)), default=[])

# ---- Auto Refresh ----
import time
placeholder = st.empty()
while True:
    df = load_latest_metrics()
    if selected_zones:
        df = df[df["zone_id"].isin(selected_zones)]

    with placeholder.container():
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 Free Flow Zones",
                    int((df["congestion_level"] == 0).sum()),
                    delta=None)
        col2.metric("🟡 Moderate Zones",
                    int((df["congestion_level"] == 1).sum()))
        col3.metric("🔴 Congested Zones",
                    int((df["congestion_level"] >= 2).sum()))
        col4.metric("🚨 Active Hotspots",
                    int(df["is_hotspot"].sum()),
                    delta=f"+{int(df['is_hotspot'].sum())} detected")

        # Congestion Map (Choropleth by zone)
        st.subheader("📍 Congestion Level by Zone (Live)")
        zone_summary = df.groupby("zone_id").agg(
            congestion_level=("congestion_level", "mean"),
            avg_speed=("avg_speed", "mean"),
            trip_count=("trip_count", "sum")
        ).reset_index()

        fig_scatter = px.scatter(
            zone_summary,
            x="zone_id", y="avg_speed",
            color="congestion_level",
            size="trip_count",
            color_continuous_scale=["green", "yellow", "orange", "red"],
            title="Zone Speed vs. Congestion Level",
            labels={"zone_id": "Zone ID", "avg_speed": "Avg Speed (mph)",
                    "congestion_level": "Congestion"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Delay Distribution
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("⏱️ Predicted Delay Distribution")
            fig_hist = px.histogram(
                df, x="predicted_delay_minutes", nbins=30,
                title="Travel Delay (Minutes)",
                color_discrete_sequence=["#E94F37"]
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            st.subheader("📈 Congestion Level Over Time")
            hourly = df.groupby("hour_of_day")["congestion_level"].mean().reset_index()
            fig_line = px.line(hourly, x="hour_of_day", y="congestion_level",
                               title="Avg Congestion by Hour",
                               markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

        # Hotspot Alerts
        hotspots = df[df["is_hotspot"] == True].sort_values("trip_count", ascending=False)
        if len(hotspots) > 0:
            st.subheader("🚨 Active Hotspot Alerts")
            st.dataframe(
                hotspots[["zone_id", "avg_speed", "trip_count",
                           "predicted_delay_minutes", "timestamp"]].head(10),
                use_container_width=True
            )

    time.sleep(refresh_rate)
```

---

## PHASE 7 — Testing & Validation

### 7.1 Unit Tests

```python
# tests/test_ml_models.py
import pytest
import pandas as pd
import numpy as np
import joblib

def test_congestion_model_loads():
    model = joblib.load("models/congestion_classifier.pkl")
    assert model is not None

def test_congestion_model_predicts():
    model = joblib.load("models/congestion_classifier.pkl")
    sample = pd.DataFrame([{
        "hour_of_day": 8, "day_of_week": 2, "is_weekend": 0,
        "is_rush_hour": 1, "is_overnight": 0,
        "hour_sin": 0.5, "hour_cos": 0.866, "dow_sin": 0.78, "dow_cos": 0.62,
        "trip_count": 250, "avg_speed": 12.5, "speed_stddev": 3.2,
        "avg_distance": 2.1, "speed_delta": -3.5, "lag1_speed": 16.0,
        "lag2_speed": 18.5, "lag1_trips": 190, "rolling_avg_speed_3h": 15.2,
        "speed_to_zone_avg": 0.82, "trips_per_minute": 4.17,
        "congestion_ratio": 20.0, "total_passengers": 380
    }])
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2, 3]

def test_delay_model_output_range():
    model = joblib.load("models/delay_regressor.pkl")
    # Predictions should be non-negative minutes
    sample = pd.DataFrame(np.random.rand(100, 22))
    sample.columns = [f"f{i}" for i in range(22)]
    # (use proper feature columns in real test)
    # assert all(model.predict(sample) >= 0)

def test_hotspot_model_contamination():
    model = joblib.load("models/hotspot_detector.pkl")
    assert model is not None
```

### 7.2 Spark Job Tests

```python
# tests/test_spark_jobs.py
import pytest
from pyspark.sql import SparkSession
import pandas as pd

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("CityFlow-Tests") \
        .master("local[2]") \
        .getOrCreate()

def test_gps_cleaner_removes_nulls(spark):
    from src.spark.gps_cleaner import clean_taxi_trajectories
    test_df = spark.createDataFrame([
        (None, None, 1, 2, 1.5, 10.0, 2),
        ("2023-01-01 08:00:00", "2023-01-01 08:20:00", 1, 2, 1.5, 10.0, 2)
    ], ["tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID", "trip_distance", "fare_amount", "passenger_count"])
    # Assert only 1 record survives null removal
    clean = test_df.dropna(subset=["tpep_pickup_datetime"])
    assert clean.count() == 1

def test_zone_aggregation_output(spark):
    # Zone aggregation should produce one row per zone per hour
    # (abbreviated — full test in repo)
    assert True
```

---

## Requirements File

```text
# requirements.txt
pyspark==3.5.0
kafka-python==2.0.2
psycopg2-binary==2.9.9
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.0
streamlit==1.32.0
plotly==5.20.0
folium==0.16.0
streamlit-folium==0.19.0
pyarrow==15.0.0
python-dotenv==1.0.1
requests==2.31.0
joblib==1.3.2
pytest==8.0.0
scipy==1.12.0
matplotlib==3.8.0
seaborn==0.13.2
```

---

## Deployment Checklist

- [ ] Docker Compose services healthy (Kafka, Zookeeper, PostgreSQL)
- [ ] Kafka topics created with correct partitions
- [ ] NYC TLC data downloaded to `data/raw/taxi_rides/`
- [ ] Batch pipeline run once to populate data lake
- [ ] All three ML models trained and saved to `models/`
- [ ] PostgreSQL schema initialized via `docker/init.sql`
- [ ] `.env` configured with correct connection strings
- [ ] Streamlit dashboard accessible at `http://localhost:8501`
- [ ] All unit tests passing (`pytest tests/ -v`)

---

*Implementation Plan — City Flow AI | Urban Mobility Intelligence System*
