# src/ingestion/kafka_producer.py
import json
import time
import random
from datetime import datetime
try:
    from kafka import KafkaProducer
except ImportError:
    print("kafka-python not installed. KafkaProducer will not be available.")
    KafkaProducer = None

try:
    from .sensor_simulator import simulate_sensor_data
except ImportError:
    from sensor_simulator import simulate_sensor_data

KAFKA_BOOTSTRAP = "localhost:9092"
TOPICS = {
    "taxi_trips": "city.taxi.trips",
    "traffic_sensors": "city.traffic.sensors",
    "weather": "city.weather.updates"
}

def get_producer():
    if KafkaProducer is None:
        return None
    try:
        return KafkaProducer(
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
    except Exception as e:
        print(f"Failed to create Kafka producer: {e}")
        return None

def stream_taxi_trips(producer):
    """Simulate streaming ride-hailing trip starts."""
    if producer is None: return
    zones = list(range(1, 264))
    print("Starting taxi trip stream...")
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

def stream_sensor_readings(producer):
    """Stream traffic sensor readings."""
    if producer is None: return
    print("Starting sensor readings stream...")
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
    producer = get_producer()
    if producer:
        t1 = threading.Thread(target=stream_taxi_trips, args=(producer,), daemon=True)
        t2 = threading.Thread(target=stream_sensor_readings, args=(producer,), daemon=True)
        t1.start()
        t2.start()
        print("✅ Kafka producers running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            producer.flush()
            producer.close()
    else:
        print("Kafka producer not started. Check if Kafka is running and kafka-python is installed.")
