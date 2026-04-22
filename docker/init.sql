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

CREATE TABLE stream_zone_windows (
    id SERIAL PRIMARY KEY,
    zone_id INT NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    trip_count_5min INT,
    avg_distance_5min FLOAT,
    total_passengers_5min INT,
    created_at TIMESTAMP DEFAULT NOW()
);
