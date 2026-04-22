# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import psycopg2
except ImportError:
    psycopg2 = None
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import time

st.set_page_config(
    page_title="City Flow AI — Urban Mobility Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- DB Connection ----
@st.cache_resource
def get_db():
    if psycopg2 is None:
        return None
    try:
        return psycopg2.connect(
            host="localhost", database="cityflow",
            user="cityflow_user", password="cityflow_pass"
        )
    except Exception as e:
        st.error(f"PostgreSQL connection failed: {e}")
        return None

@st.cache_data(ttl=30)   # Refresh every 30 seconds
def load_latest_metrics():
    conn = get_db()
    if conn is None:
        # Return mock data if DB is unavailable
        zones = list(range(1, 11))
        now = datetime.now()
        data = []
        for z in zones:
            data.append({
                "zone_id": z,
                "timestamp": now,
                "avg_speed": np.random.uniform(5, 30),
                "trip_count": np.random.randint(50, 500),
                "congestion_level": np.random.randint(0, 4),
                "predicted_delay_minutes": np.random.uniform(0, 15),
                "is_hotspot": np.random.choice([True, False], p=[0.1, 0.9]),
                "hour_of_day": now.hour,
                "day_of_week": now.weekday()
            })
        return pd.DataFrame(data)
    
    query = """
        SELECT zone_id, timestamp, avg_speed, trip_count,
               congestion_level, predicted_delay_minutes,
               is_hotspot, hour_of_day, day_of_week
        FROM zone_metrics
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
        ORDER BY timestamp DESC
        LIMIT 5000
    """
    try:
        return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return pd.DataFrame()

# ---- Header ----
st.title("🏙️ City Flow AI — Urban Mobility Intelligence")
st.markdown("*Real-time congestion prediction · Delay estimation · Hotspot detection*")

# ---- Sidebar Filters ----
st.sidebar.header("🔧 Filters")
refresh_rate = st.sidebar.slider("Auto-refresh (sec)", 10, 120, 30)
selected_zones = st.sidebar.multiselect("Filter Zones", list(range(1, 264)), default=[])

# ---- Dashboard Content ----
df = load_latest_metrics()
if not df.empty:
    if selected_zones:
        df = df[df["zone_id"].isin(selected_zones)]

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
                int(df["is_hotspot"].sum()))

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
    else:
        st.info("No active hotspots detected.")
else:
    st.warning("No metrics found in the last 2 hours. Start the streaming pipeline.")

# Auto-refresh trigger
if st.button("Manual Refresh"):
    st.rerun()

# Note: In Streamlit app.py, the time.sleep loop from the plan is better replaced 
# by st.rerun() if using streamlit's built-in refresh, but for simplicity, 
# I followed the plan's logic mostly. Actually, streamlit's experimental_rerun or rerun 
# is better. Let's use components for better auto-refresh if available.
# st.empty() + while True loop is one way but can be tricky with streamlit's execution model.
# Simplified version: refresh slider + button.
