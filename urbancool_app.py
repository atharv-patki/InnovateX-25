import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import time
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="UrbanCool AI Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- App Styling ---
st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
        }
        .stMetric {
            border: 1px solid #2e3b4d;
            background-color: #1a222f;
            border-radius: 10px;
            padding: 15px;
        }
        .stMetric .st-ae {
            color: #4f8bf9; /* Label color */
        }
    </style>
""", unsafe_allow_html=True)


# --- Data Simulation ---
@st.cache_data
def get_city_data(city_name):
    """Generates static data for a selected city."""
    if city_name == "Delhi":
        return {
            "center": [28.6139, 77.2090],
            "zoom": 12,
            "zones": {f"Zone {i}": (28.61 + np.random.uniform(-0.05, 0.05), 77.20 + np.random.uniform(-0.05, 0.05)) for i in range(1, 11)},
            "devices": {
                "Misting Fountain 1": (28.62, 77.21), "Shade Sail 1": (28.63, 77.22), "Solar Fan 1": (28.60, 77.19),
                "Misting Fountain 2": (28.59, 77.23), "Shade Sail 2": (28.64, 77.18)
            }
        }
    elif city_name == "Mumbai":
        return {
            "center": [19.0760, 72.8777],
            "zoom": 12,
            "zones": {f"Zone {i}": (19.07 + np.random.uniform(-0.05, 0.05), 72.87 + np.random.uniform(-0.05, 0.05)) for i in range(1, 11)},
            "devices": {
                "Misting Fountain 1": (19.08, 72.88), "Shade Sail 1": (19.06, 72.86), "Solar Fan 1": (19.09, 72.85),
                "Misting Fountain 2": (19.05, 72.89), "Shade Sail 2": (19.07, 72.90)
            }
        }
    else: # Bangalore
        return {
            "center": [12.9716, 77.5946],
            "zoom": 12,
            "zones": {f"Zone {i}": (12.97 + np.random.uniform(-0.05, 0.05), 77.59 + np.random.uniform(-0.05, 0.05)) for i in range(1, 11)},
            "devices": {
                "Misting Fountain 1": (12.98, 77.60), "Shade Sail 1": (12.96, 77.58), "Solar Fan 1": (12.95, 77.61),
                "Misting Fountain 2": (12.99, 77.57), "Shade Sail 2": (12.97, 77.62)
            }
        }

def simulate_live_data(temp_base, hum_base, hi_base, crowd_sensors):
    """Simulates real-time changing data points."""
    temp = temp_base + np.random.uniform(-1.5, 1.5)
    humidity = hum_base + np.random.uniform(-5, 5)
    heat_index = hi_base + np.random.uniform(-2, 2)
    
    crowd_data = {
        "Bus Stops": random.randint(20, 100) if crowd_sensors else 0,
        "Parks": random.randint(50, 300) if crowd_sensors else 0,
        "Plazas": random.randint(100, 500) if crowd_sensors else 0,
    }
    return temp, humidity, heat_index, crowd_data

def run_ai_prediction(temp, humidity, heat_index, crowd_data, sensitivity, city_data):
    """Simulates the AI model predicting high-risk zones."""
    zone_predictions = {}
    for zone, coords in city_data["zones"].items():
        # Factors: high temp, high humidity, high crowd numbers
        temp_score = (temp - 30) / 10
        hum_score = (humidity - 60) / 40
        crowd_score = sum(crowd_data.values()) / 900 # Max crowd = 900
        
        # Randomness to simulate geographical variations
        zone_factor = np.random.uniform(0.8, 1.2)
        
        # Combine scores with sensitivity adjustment
        risk_score = (temp_score + hum_score + crowd_score) * sensitivity * zone_factor
        zone_predictions[zone] = max(0, min(1, risk_score)) # Normalize between 0 and 1
        
    return zone_predictions

# --- UI Components ---
def create_heatmap(predictions):
    """Creates a Plotly heatmap for AI predictions."""
    zones = list(predictions.keys())
    scores = list(predictions.values())
    colorscale = [[0, 'rgb(0,100,255)'], [0.5, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']]
    
    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        x=zones,
        y=['Risk Score'],
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        showscale=False))
    
    fig.update_layout(
        title="AI Prediction: Hot + Crowded Zones",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white"
    )
    return fig

def create_device_map(city_data, predictions, devices_status):
    """Creates a Folium map with dynamic device markers."""
    m = folium.Map(location=city_data["center"], zoom_start=city_data["zoom"], tiles="CartoDB positron")

    # Add zone markers
    for zone, score in predictions.items():
        color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
        folium.CircleMarker(
            location=city_data["zones"][zone],
            radius=5,
            popup=f"{zone}: Risk Score {score:.2f}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Add device markers
    for device, coords in city_data["devices"].items():
        status = devices_status[device]["status"]
        intensity = devices_status[device]["intensity"]
        icon_color = "green" if status == "ON" else "red"
        icon_type = "tint" if "Misting" in device else "umbrella-beach" if "Shade" in device else "fan"
        
        popup_html = f"""
        <b>{device}</b><br>
        Status: <b style='color:{icon_color};'>{status}</b><br>
        Intensity: {intensity}%
        """
        
        folium.Marker(
            location=coords,
            popup=popup_html,
            icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
        ).add_to(m)
        
    return m

def create_feedback_charts(history):
    """Creates charts for energy and temperature feedback."""
    df = pd.DataFrame(history)
    
    # Energy Consumption
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(x=df['time'], y=df['baseline_energy'], mode='lines', name='Baseline Schedule', line=dict(dash='dot', color='orange')))
    fig_energy.add_trace(go.Scatter(x=df['ai_energy'], mode='lines', name='UrbanCool AI', fill='tozeroy', line=dict(color='#4f8bf9')))
    fig_energy.update_layout(
        title="Energy Consumption (kW)",
        xaxis_title="Time",
        yaxis_title="Power",
        legend=dict(x=0, y=1.1, orientation='h'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white"
    )
    
    # Temperature Drop
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df['time'], y=df['temp_drop'], mode='lines', name='Temp Drop', line=dict(color='lightgreen')))
    fig_temp.update_layout(
        title="Temperature Drop in Cooled Zones (°C)",
        xaxis_title="Time",
        yaxis_title="Δ Temperature",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white"
    )
    
    return fig_energy, fig_temp

# --- Main App ---
def main():
    st.title("❄️ UrbanCool: AI-Driven Weather-Responsive Cooling System")
    st.markdown("##### *Cooling Smarter. Living Better.*")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("System Controls")
        selected_city = st.selectbox("Select City", ["Delhi", "Mumbai", "Bangalore"])
        
        st.subheader("Simulate Weather Forecast")
        temp_base = st.slider("Temperature (°C)", 25.0, 45.0, 35.0, 0.5)
        hum_base = st.slider("Humidity (%)", 20, 90, 60, 5)
        hi_base = st.slider("Heat Index", 30.0, 50.0, 40.0, 0.5)
        
        st.subheader("Simulate IoT Sensors")
        crowd_sensors = st.toggle("Enable Crowd Sensors", True)
        
        st.subheader("AI Configuration")
        ai_sensitivity = st.slider("AI Prediction Aggressiveness", 0.5, 2.0, 1.0, 0.1)

    # --- Initializations ---
    city_data = get_city_data(selected_city)
    if 'history' not in st.session_state or st.session_state.city != selected_city:
        st.session_state.history = []
        st.session_state.city = selected_city
        st.session_state.anomaly = None

    # --- Real-time Simulation Loop ---
    # Simulate data
    temp, humidity, heat_index, crowd_data = simulate_live_data(temp_base, hum_base, hi_base, crowd_sensors)
    
    # AI Prediction
    zone_predictions = run_ai_prediction(temp, humidity, heat_index, crowd_data, ai_sensitivity, city_data)
    
    # Anomaly Simulation
    if random.random() < 0.05: # 5% chance of anomaly
        st.session_state.anomaly = random.choice(["Sensor Failure", "Network Delay"])
    elif random.random() < 0.2: # 20% chance of recovery
        st.session_state.anomaly = None

    if st.session_state.anomaly:
        st.warning(f"🚨 Anomaly Detected: **{st.session_state.anomaly}**. Switching to default safe cooling mode.")
        # In fallback mode, AI predictions are ignored, devices run on a safe, lower-intensity schedule.
        for zone in zone_predictions:
            zone_predictions[zone] = 0.3 # Dampen predictions

    # Device Control Logic
    devices_status = {}
    active_devices = 0
    for device in city_data["devices"]:
        # A simple logic: activate device if it's near a high-risk zone
        device_lat, device_lon = city_data["devices"][device]
        is_near_hotspot = any(
            np.sqrt((device_lat - zone_lat)**2 + (device_lon - zone_lon)**2) < 0.02 and score > 0.6
            for zone, (zone_lat, zone_lon) in city_data["zones"].items()
            for score in [zone_predictions[zone]]
        )
        
        if is_near_hotspot:
            devices_status[device] = {"status": "ON", "intensity": random.randint(50, 100)}
            active_devices += 1
        else:
            devices_status[device] = {"status": "OFF", "intensity": 0}

    # Feedback Loop Data
    baseline_energy = 5 * len(city_data["devices"]) # Assume baseline is all devices at 100%
    ai_energy = sum(d['intensity'] / 100 * 5 for d in devices_status.values()) # 5kW per device max
    temp_drop = (ai_energy / baseline_energy) * 2.5 * np.random.uniform(0.9, 1.1) if baseline_energy > 0 else 0

    # Update history
    current_time = pd.Timestamp.now()
    st.session_state.history.append({
        'time': current_time,
        'baseline_energy': baseline_energy,
        'ai_energy': ai_energy,
        'temp_drop': temp_drop
    })
    if len(st.session_state.history) > 50: # Keep history to last 50 points
        st.session_state.history.pop(0)

    # --- Dashboard Layout ---
    
    # Row 1: Data Fusion Panel
    st.subheader("Data Fusion Panel")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🌡️ Weather Forecast", value=f"{temp:.1f}°C", delta=f"Feels like {heat_index:.1f}°C")
    with col2:
        avg_risk = np.mean(list(zone_predictions.values()))
        st.metric(label="🗺️ Heat Island Index", value=f"{avg_risk:.2f}", help="Average risk score across all zones.")
    with col3:
        total_people = sum(crowd_data.values())
        st.metric(label="👥 Human Presence", value=f"{total_people} Est.", delta="Real-time IoT Counts")

    # Row 2: AI Prediction & Device Control
    st.subheader("AI Prediction & Control Simulation")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.plotly_chart(create_heatmap(zone_predictions), use_container_width=True)
    with col2:
        st_folium(create_device_map(city_data, zone_predictions, devices_status), use_container_width=True, height=400)
    
    # Row 3: Feedback Loop
    st.subheader("System Feedback Loop")
    if st.session_state.history:
        fig_energy, fig_temp = create_feedback_charts(st.session_state.history)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_energy, use_container_width=True)
        with col2:
            st.plotly_chart(fig_temp, use_container_width=True)

    # Row 4: Benefits Panel
    st.subheader("Performance & Benefits")
    energy_saved_pct = (1 - ai_energy / baseline_energy) * 100 if baseline_energy > 0 else 0
    comfort_index = temp_drop / (total_people + 1) * 1000 # Arbitrary metric
    grid_demand_score = (1 - np.std([h['ai_energy'] for h in st.session_state.history]) / baseline_energy) * 100 if len(st.session_state.history) > 1 else 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="⚡ Energy Saved", value=f"{energy_saved_pct:.1f}%", delta="Compared to static schedule")
    with col2:
        st.metric(label="😊 Avg. Comfort Index", value=f"{comfort_index:.2f}", help="Temperature reduction per person")
    with col3:
        st.metric(label="📉 Grid Demand Smoothness", value=f"{grid_demand_score:.1f}/100", delta="Lower is volatile")


if __name__ == "__main__":
    main()
    # Auto-refresh the page every 3 seconds for a real-time feel
    time.sleep(3)
    st.rerun()
