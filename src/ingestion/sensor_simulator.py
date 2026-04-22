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
