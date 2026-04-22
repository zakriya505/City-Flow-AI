# 🏙️ City Flow AI — Urban Mobility Intelligence System

City Flow AI is a production-grade big data and machine learning system that models real-time urban mobility behavior. It ingests millions of taxi ride events and traffic sensor readings, processes them through Apache Spark at scale, and applies three distinct ML models to provide actionable intelligence about city traffic conditions.

## 🚀 System Architecture

The system follows a modern data engineering architecture:
- **Ingestion**: Apache Kafka handles real-time streams of taxi trips and sensor data.
- **Batch Processing**: Apache Spark processes historical NYC TLC data for training and long-term analysis.
- **Streaming**: Spark Structured Streaming processes live data with sliding windows and watermarking.
- **Storage**: Apache Parquet for the data lake and PostgreSQL for the serving layer.
- **ML Pipeline**: Scikit-learn models for congestion classification, delay estimation, and hotspot detection.
- **Visualization**: A real-time Streamlit dashboard for monitoring city-wide traffic.

## 🛠️ Tech Stack

- **Data Processing**: Apache Spark 3.5, PySpark
- **Streaming**: Apache Kafka 3.6
- **Database**: PostgreSQL 15
- **Machine Learning**: Scikit-learn 1.4, Joblib
- **Dashboard**: Streamlit 1.32, Plotly
- **Storage**: Apache Parquet, Snappy Compression
- **Environment**: Docker, Python 3.10+

## 📁 Project Structure

```
City-Flow-AI/
├── data/
│   ├── raw/                # Raw ingestion (Taxi rides, Sensors)
│   └── lake/               # Cleaned Parquet data lake
├── docker/                 # Infrastructure configuration
│   └── init.sql            # Database schema
├── models/                 # Serialized ML models
├── src/
│   ├── ingestion/          # Kafka producers & simulators
│   ├── spark/              # Batch & Streaming pipelines
│   ├── ml/                 # Feature engineering & ML training
│   └── dashboard/          # Streamlit visualization app
├── tests/                  # Unit and integration tests
├── docker-compose.yml      # Infrastructure orchestration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚡ Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Java 11 (for Spark)
- Python 3.10+

### 2. Set Up Infrastructure
```bash
docker-compose up -d
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
1. **Download Data**: `python src/ingestion/data_downloader.py`
2. **Batch Process**: `python src/spark/batch_processing.py`
3. **Train Models**: `python src/ml/train_models.py`
4. **Start Streaming**: `python src/spark/streaming_pipeline.py`
5. **Start Producer**: `python src/ingestion/kafka_producer.py`
6. **Launch Dashboard**: `streamlit run src/dashboard/app.py`

## 📊 Machine Learning Tasks

1. **Congestion Classification**: Predicts one of 4 levels (Free, Moderate, Heavy, Gridlock) for each city zone.
2. **Delay Estimation**: Predicts expected travel delay in minutes using Gradient Boosting Regression.
3. **Hotspot Detection**: Unsupervised anomaly detection using Isolation Forest to identify emerging traffic bottlenecks.

## 📈 Performance & Optimization

- **Spark AQE**: Adaptive Query Execution enabled for handling NYC data skew.
- **Watermarking**: 2-minute watermarking in streaming to handle late-arriving IoT data.
- **Indexing**: Optimized PostgreSQL composite and partial indexes for sub-20ms dashboard queries.
- **Compression**: Snappy codec for Parquet storage, achieving ~75% I/O reduction.

## 📜 License
This project is licensed under the MIT License.
