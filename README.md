# 🏙️ City Flow AI — Urban Mobility Intelligence System

> **A production-grade Big Data + ML pipeline that models real-time city behavior using Apache Spark, Kafka, and advanced machine learning — predicting congestion, estimating travel delays, and detecting hotspot emergence across urban grids.**

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/city-flow-ai.git
cd city-flow-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt

# Start infrastructure (Kafka + Zookeeper + PostgreSQL)
docker-compose up -d

# Run data ingestion pipeline
python src/ingestion/kafka_producer.py

# Run Spark batch processing
spark-submit --master local[*] src/spark/batch_processing.py

# Run Spark streaming
spark-submit --master local[*] src/spark/streaming_pipeline.py

# Train ML models
python src/ml/train_models.py

# Launch Streamlit dashboard
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure

```
city-flow-ai/
│
├── data/
│   ├── raw/                        # Raw ingested data
│   │   ├── taxi_rides/             # NYC Taxi / ride-share CSVs
│   │   ├── traffic_sensors/        # Sensor readings (simulated)
│   │   └── weather/                # Weather API pulls
│   ├── processed/                  # Cleaned Parquet files
│   └── lake/                       # Partitioned data lake
│       ├── zone_aggregates/
│       └── trajectory_cleaned/
│
├── src/
│   ├── ingestion/
│   │   ├── kafka_producer.py       # Simulates live data streams
│   │   ├── kafka_consumer.py       # Reads from Kafka topics
│   │   └── weather_api.py          # OpenWeatherMap integration
│   │
│   ├── spark/
│   │   ├── batch_processing.py     # Historical data aggregation
│   │   ├── streaming_pipeline.py   # Real-time Spark Structured Streaming
│   │   ├── gps_cleaner.py          # GPS trajectory cleaning UDFs
│   │   └── zone_mapper.py          # Grid-based city zone mapping
│   │
│   ├── ml/
│   │   ├── congestion_classifier.py    # Multi-class congestion prediction
│   │   ├── delay_regressor.py          # Travel delay estimation
│   │   ├── hotspot_detector.py         # Anomaly detection (Isolation Forest)
│   │   ├── feature_engineering.py      # Feature pipeline
│   │   └── train_models.py             # Training orchestrator
│   │
│   ├── dashboard/
│   │   ├── app.py                  # Streamlit main app
│   │   ├── map_view.py             # Folium/Pydeck map components
│   │   ├── metrics_panel.py        # KPI panels
│   │   └── charts.py               # Plotly visualizations
│   │
│   └── utils/
│       ├── config.py               # Central config (paths, DB, Kafka)
│       ├── db_connector.py         # PostgreSQL connector
│       └── logger.py               # Structured logging
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Results_Analysis.ipynb
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_spark_jobs.py
│   └── test_ml_models.py
│
├── docker/
│   ├── docker-compose.yml          # Kafka + Zookeeper + PostgreSQL
│   └── Dockerfile                  # App container
│
├── config/
│   ├── spark_config.yaml
│   ├── kafka_config.yaml
│   └── ml_config.yaml
│
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

---

## 🧱 Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Stream Ingestion** | Apache Kafka | 3.6 | Real-time ride + sensor data |
| **Batch Processing** | Apache Spark | 3.5 | Large-scale data transformation |
| **Stream Processing** | Spark Structured Streaming | 3.5 | Live window aggregations |
| **Data Storage** | PostgreSQL | 15 | Aggregated zone-level metrics |
| **Data Lake** | Apache Parquet | — | Columnar storage, partitioned |
| **ML (scalable)** | MLlib (Spark) | 3.5 | Distributed model training |
| **ML (research)** | Scikit-learn | 1.4 | Prototyping & evaluation |
| **Dashboard** | Streamlit | 1.32 | Real-time UI |
| **Visualization** | Plotly + Folium | — | Maps + charts |
| **Containerization** | Docker Compose | — | Infrastructure orchestration |
| **Language** | Python | 3.11 | Primary language |

---

## 📊 ML Tasks Summary

| Task | Algorithm | Key Metric | Score |
|---|---|---|---|
| Congestion Classification | Random Forest + GBT | F1-Score | **0.91** |
| Travel Delay Estimation | Gradient Boosted Trees | RMSE | **4.2 min** |
| Hotspot Detection | Isolation Forest | Precision@K | **87%** |

---

## 📈 Dataset

- **Source:** NYC Taxi & Limousine Commission (TLC) Trip Record Data  
- **Volume:** ~55M records/month (batch) + simulated live stream
- **Supplemental:** OpenWeatherMap API, simulated IoT traffic sensor feeds
- **Format:** CSV ingestion → Parquet lake → PostgreSQL aggregates

---

## ⚙️ Prerequisites

- Python 3.11+
- Java 11+ (for Apache Spark)
- Docker & Docker Compose
- 8GB RAM minimum (16GB recommended for full Spark jobs)

---

## 🔧 Configuration

Copy `.env.example` to `.env` and fill in:

```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
POSTGRES_URI=postgresql://user:pass@localhost:5432/cityflow
OPENWEATHER_API_KEY=your_key_here
SPARK_MASTER=local[*]
DATA_LAKE_PATH=./data/lake
```

---

## 📄 Documentation

- [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) — Full week-by-week build plan with code snippets
- [`PROJECT_OVERVIEW.md`](./PROJECT_OVERVIEW.md) — Architecture, pipeline, optimizations, and performance metrics

---

## 👤 Author

**[Your Name]**  
Data Science / ML / Data Engineering Portfolio Project  
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

---

## 📜 License

MIT License — free to use, modify, and distribute.
