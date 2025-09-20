import streamlit as st
import pandas as pd
import numpy as np
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


# --- Enhanced App Styling ---
st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
        }
        
        /* Enhanced metric styling with better visibility */
        .stMetric {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
            border: 2px solid #e9ecef !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            margin: 8px 0 !important;
        }
        
        /* Metric label styling */
        .stMetric > label {
            color: #495057 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Metric value styling */
        .stMetric > div {
            color: #212529 !important;
            font-weight: 700 !important;
            font-size: 24px !important;
        }
        
        /* Metric delta styling */
        .stMetric > div > div {
            color: #6c757d !important;
            font-size: 12px !important;
        }
        
        /* Info, warning, success box improvements */
        .stAlert {
            border-radius: 8px !important;
            border-left: 4px solid !important;
            padding: 15px !important;
        }
        
        .stAlert[data-baseweb="notification"] {
            background-color: rgba(255, 255, 255, 0.95) !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #007bff !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
        }
        
        .stButton > button:hover {
            background-color: #0056b3 !important;
            transform: translateY(-2px) !important;
        }
        
        /* Chart container improvements */
        .js-plotly-plot {
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Subheader styling */
        .stMarkdown h2, .stMarkdown h3 {
            color: #2c3e50 !important;
            border-bottom: 2px solid #3498db !important;
            padding-bottom: 8px !important;
            margin-bottom: 20px !important;
        }
    </style>
""", unsafe_allow_html=True)


# --- Data Simulation ---
@st.cache_data
def get_city_data(city_name):
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
    elif city_name == "Pune":
        return {
            "center": [18.5204, 73.8567],
            "zoom": 12,
            "zones": {f"Zone {i}": (18.52 + np.random.uniform(-0.05, 0.05), 73.85 + np.random.uniform(-0.05, 0.05)) for i in range(1, 11)},
            "devices": {
                "Misting Fountain 1": (18.53, 73.86), "Shade Sail 1": (18.51, 73.84), "Solar Fan 1": (18.54, 73.87),
                "Misting Fountain 2": (18.50, 73.83), "Shade Sail 2": (18.55, 73.88)
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
    zone_predictions = {}
    for zone, coords in city_data["zones"].items():
        temp_score = (temp - 30) / 10
        hum_score = (humidity - 60) / 40
        crowd_score = sum(crowd_data.values()) / 900
        zone_factor = np.random.uniform(0.8, 1.2)
        risk_score = (temp_score + hum_score + crowd_score) * sensitivity * zone_factor
        zone_predictions[zone] = max(0, min(1, risk_score))
    return zone_predictions


# --- UI Components ---
def create_heatmap(predictions):
    zones = list(predictions.keys())
    scores = list(predictions.values())
    colorscale = [[0, 'rgb(0,100,255)'], [0.5, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']]


    fig = go.Figure(data=go.Heatmap(
        z=[scores], x=zones, y=['Risk Score'],
        colorscale=colorscale, zmin=0, zmax=1, showscale=True
    ))
    fig.update_layout(
        title="AI Prediction: Hot + Crowded Zones",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color="#2c3e50",
        title_font_size=16,
        height=300
    )
    return fig


def create_device_map(city_data, predictions, devices_status):
    m = folium.Map(location=city_data["center"], zoom_start=city_data["zoom"], tiles="CartoDB positron")
    
    for zone, score in predictions.items():
        color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
        folium.CircleMarker(
            location=city_data["zones"][zone],
            radius=8,
            popup=f"<b>{zone}</b><br>Risk Score: {score:.2f}",
            color=color, fill=True, fill_color=color, weight=2
        ).add_to(m)


    for device, coords in city_data["devices"].items():
        status = devices_status[device]["status"]
        intensity = devices_status[device]["intensity"]
        icon_color = "green" if status == "ON" else "red"
        icon_type = "tint" if "Misting" in device else "umbrella-beach" if "Shade" in device else "fan"
        popup_html = f"""
        <div style='width:200px'>
            <b style='color: #2c3e50;'>{device}</b><br>
            Status: <b style='color:{icon_color};'>{status}</b><br>
            Intensity: <b>{intensity}%</b>
        </div>
        """
        folium.Marker(
            location=coords,
            popup=popup_html,
            icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
        ).add_to(m)
    return m


def create_feedback_charts(history):
    df = pd.DataFrame(history)
    
    # Energy chart with improved styling
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(
        x=df['time'], y=df['baseline_energy'], 
        mode='lines', name='Baseline Schedule', 
        line=dict(dash='dot', color='#e74c3c', width=3)
    ))
    fig_energy.add_trace(go.Scatter(
        x=df['time'], y=df['ai_energy'], 
        mode='lines', name='UrbanCool AI', 
        fill='tozeroy', line=dict(color='#3498db', width=3)
    ))
    fig_energy.update_layout(
        title="Energy Consumption (kW)", 
        xaxis_title="Time", 
        yaxis_title="Power (kW)",
        legend=dict(x=0, y=1.1, orientation='h'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color="#2c3e50",
        height=400
    )


    # Temperature chart with improved styling
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df['time'], y=df['temp_drop'], 
        mode='lines+markers', name='Temp Drop', 
        line=dict(color='#27ae60', width=3),
        marker=dict(size=6)
    ))
    fig_temp.update_layout(
        title="Temperature Drop in Cooled Zones (°C)", 
        xaxis_title="Time", 
        yaxis_title="Δ Temperature (°C)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color="#2c3e50",
        height=400
    )
    return fig_energy, fig_temp


# --- Main App ---
def main():
    st.title("❄️ UrbanCool: AI-Driven Weather-Responsive Cooling System")
    st.markdown("##### *Cooling Smarter. Living Better.*")


    # Auto-refresh control
    with st.sidebar:
        st.header("🔄 Refresh Controls")
        auto_refresh = st.toggle("Enable Auto Refresh", False)
        if auto_refresh:
            refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1)
        
        # Manual refresh button
        if st.button("🔄 Refresh Data Now"):
            st.rerun()


    # Intro Card
    st.subheader("🚀 Introducing UrbanCool")
    st.info("UrbanCool is an AI-driven orchestration system that fuses weather forecasts, heat island maps, and human presence metrics to deliver **targeted, efficient cooling** across cities. It reduces wasted energy, improves comfort, and ensures safer public spaces.")


    # Problem Context
    st.subheader("🌍 The Urban Heat Island Problem")
    st.markdown("Urban environments trap heat, creating **heat islands** that raise local temperatures by several degrees, wasting energy, raising costs, and reducing comfort.")


    # Sidebar Controls
    with st.sidebar:
        st.header("System Controls")
        selected_city = st.selectbox("Select City", ["Delhi", "Mumbai", "Pune", "Bangalore"])
        
        st.subheader("Simulate Weather Forecast")
        temp_base = st.slider("Temperature (°C)", 25.0, 45.0, 35.0, 0.5)
        hum_base = st.slider("Humidity (%)", 20, 90, 60, 5)
        hi_base = st.slider("Heat Index", 30.0, 50.0, 40.0, 0.5)
        
        st.subheader("Simulate IoT Sensors")
        crowd_sensors = st.toggle("Enable Crowd Sensors", True)
        
        st.subheader("AI Configuration")
        ai_sensitivity = st.slider("AI Prediction Aggressiveness", 0.5, 2.0, 1.0, 0.1)
        
        st.subheader("Future-Proofing")
        scale_factor = st.slider("Neighborhoods Covered", 1, 10, 1)
        new_device = st.selectbox("Add New Device Type", ["None", "Evaporative Bench", "Cooling Tunnel"])


    city_data = get_city_data(selected_city)
    if new_device != "None":
        city_data["devices"][f"{new_device} 1"] = (
            city_data["center"][0] + np.random.uniform(-0.03, 0.03),
            city_data["center"][1] + np.random.uniform(-0.03, 0.03)
        )


    # Initialize session state
    if 'history' not in st.session_state or st.session_state.get('city') != selected_city:
        st.session_state.history = []
        st.session_state.city = selected_city
        st.session_state.anomaly = None


    # Simulate data
    temp, humidity, heat_index, crowd_data = simulate_live_data(temp_base, hum_base, hi_base, crowd_sensors)
    zone_predictions = run_ai_prediction(temp, humidity, heat_index, crowd_data, ai_sensitivity, city_data)


    # Anomaly simulation
    if random.random() < 0.05:
        st.session_state.anomaly = random.choice(["Sensor Failure", "Network Delay"])
    elif random.random() < 0.2:
        st.session_state.anomaly = None


    if st.session_state.anomaly:
        st.warning(f"🚨 Anomaly Detected: **{st.session_state.anomaly}**. Switching to safe mode.")
        for zone in zone_predictions:
            zone_predictions[zone] = 0.3


    # Device status simulation
    devices_status = {}
    for device in city_data["devices"]:
        device_lat, device_lon = city_data["devices"][device]
        is_near_hotspot = any(
            np.sqrt((device_lat - zone_lat)**2 + (device_lon - zone_lon)**2) < 0.02 and score > 0.6
            for zone, (zone_lat, zone_lon) in city_data["zones"].items()
            for score in [zone_predictions[zone]]
        )
        if is_near_hotspot:
            devices_status[device] = {"status": "ON", "intensity": random.randint(50, 100)}
        else:
            devices_status[device] = {"status": "OFF", "intensity": 0}


    # Energy calculations
    baseline_energy = 5 * len(city_data["devices"])
    ai_energy = sum(d['intensity'] / 100 * 5 for d in devices_status.values())
    temp_drop = (ai_energy / baseline_energy) * 2.5 * np.random.uniform(0.9, 1.1) if baseline_energy > 0 else 0


    # Update history
    current_time = pd.Timestamp.now()
    st.session_state.history.append({
        'time': current_time, 
        'baseline_energy': baseline_energy, 
        'ai_energy': ai_energy, 
        'temp_drop': temp_drop
    })
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)


    # --- Dashboard Layout ---
    st.subheader("📊 Data Fusion Panel")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡️ Weather Forecast", f"{temp:.1f}°C", delta=f"Feels like {heat_index:.1f}°C")
    with col2:
        avg_risk = np.mean(list(zone_predictions.values()))
        st.metric("🗺️ Heat Island Index", f"{avg_risk:.2f}", help="Average risk across all zones")
    with col3:
        total_people = sum(crowd_data.values())
        st.metric("👥 Human Presence", f"{total_people} Est.", delta="Real-time IoT Counts")


    # AI Prediction & Device Control
    st.subheader("🤖 AI Prediction & Control Simulation")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.plotly_chart(create_heatmap(zone_predictions), use_container_width=True)
    with col2:
        st_folium(create_device_map(city_data, zone_predictions, devices_status), 
                 use_container_width=True, height=400)


    # Feedback Loop
    st.subheader("🔄 System Feedback Loop")
    st.info("🔄 AI model retrains every 5 minutes using live data on energy usage and temperature drops.")
    
    if st.session_state.history:
        fig_energy, fig_temp = create_feedback_charts(st.session_state.history)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_energy, use_container_width=True)
        with col2:
            st.plotly_chart(fig_temp, use_container_width=True)


    # Benefits Panel
    st.subheader("📈 Performance & Benefits")
    energy_saved_pct = (1 - ai_energy / baseline_energy) * 100 if baseline_energy > 0 else 0
    comfort_index = temp_drop / (total_people + 1) * 1000
    safe_zones = sum(1 for score in zone_predictions.values() if score < 0.4)
    health_risk_reduction = safe_zones * 10


    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("⚡ Energy Saved", f"{energy_saved_pct:.1f}%", delta="vs static schedule")
    with col2:
        st.metric("💰 Cost Savings", f"₹{ai_energy*5:.0f}/hr", help="Estimated reduction in electricity cost")
    with col3:
        st.metric("😊 Comfort Index", f"{comfort_index:.2f}", help="Temperature reduction per person")
    with col4:
        st.metric("✅ Safe Zones", f"{safe_zones}/10", help="Zones under safe heat risk")
    with col5:
        st.metric("🛡️ Health Risk Reduction", f"{health_risk_reduction}%", help="Estimated reduction in heat illness risk")


    # Grid Resilience
    st.subheader("📉 Grid Resilience")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig_grid = go.Figure()
        fig_grid.add_trace(go.Scatter(
            x=df['time'], y=df['baseline_energy'], 
            mode='lines', name='Baseline Demand',
            line=dict(color='#e74c3c', width=3)
        ))
        fig_grid.add_trace(go.Scatter(
            x=df['time'], y=df['ai_energy'], 
            mode='lines', name='AI-Controlled Demand',
            line=dict(color='#3498db', width=3)
        ))
        fig_grid.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color="#2c3e50",
            height=400
        )
        st.plotly_chart(fig_grid, use_container_width=True)


    # System Status
    st.subheader("⚠️ System Alerts & Maintenance")
    if st.session_state.anomaly:
        st.error(f"🚨 Anomaly detected: {st.session_state.anomaly}")
    else:
        st.success("✅ All systems functional")


    st.warning("🔒 IoT Security: All commands authenticated. Current latency: 120ms")
    
    with st.expander("🛠️ Device Maintenance Log"):
        for device in devices_status:
            service_hours = random.randint(10, 100)
            status_emoji = "🟢" if devices_status[device]["status"] == "ON" else "🔴"
            st.write(f"{status_emoji} **{device}** – {service_hours} hrs since last service")


    # Future-Proofing
    st.subheader("🚀 Future-Proofing Simulation")
    projected_energy_saving = energy_saved_pct * scale_factor
    projected_safe_zones = safe_zones * scale_factor
    co2_saved = energy_saved_pct * 0.8


    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌍 Projected Energy Savings", f"{projected_energy_saving:.1f}%", 
                 help=f"Across {scale_factor} neighborhoods")
    with col2:
        st.metric("🏙️ Projected Safe Zones", f"{projected_safe_zones} total")
    with col3:
        st.metric("🌱 Estimated CO₂ Saved", f"{co2_saved:.1f} kg/hr")


    # Auto-refresh logic (only if enabled)
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
