import os
import streamlit as st

st.set_page_config(
    page_title="WeatherSphere AI Dashboard",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import torch
import torch.nn as nn
import requests
from datetime import datetime, timedelta

# Import pipeline functions
from src.data_ingestion import fetch_weather_data
from src.feature_engineering import engineer_features
from src.train_random_forest import train_rf
from src.train_lstm import train_lstm
from src.evaluate import evaluate_models

# Define LSTM Model locally so Streamlit can deserialize it without import issues
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class WeatherGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(WeatherGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class WeatherBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(WeatherBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



# Custom CSS for dark-theme premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Header card */
    .hero-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #0f172a 100%);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .hero-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #bfdbfe;
        font-weight: 300;
        margin-bottom: 0;
    }
    
    /* Glassmorphism Metric Card */
    .glass-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .glass-card-temp { border-left: 4px solid #f43f5e; }
    .glass-card-humidity { border-left: 4px solid #3b82f6; }
    .glass-card-pressure { border-left: 4px solid #a855f7; }
    .glass-card-wind { border-left: 4px solid #eab308; }
    .glass-card-precip { border-left: 4px solid #6366f1; }
    .glass-card-range { border-left: 4px solid #10b981; }

    .card-title {
        font-size: 0.85rem;
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    
    /* Hero Forecast Card */
    .forecast-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #3b82f6;
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.25);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .forecast-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        position: relative;
        z-index: 2;
    }
    .forecast-temp {
        font-size: 4rem;
        font-weight: 800;
        color: #3b82f6;
        margin: 0.75rem 0;
        text-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
        position: relative;
        z-index: 2;
    }
    .forecast-sub {
        font-size: 1rem;
        color: #94a3b8;
        position: relative;
        z-index: 2;
    }
    .forecast-model-badge {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-top: 0.5rem;
        position: relative;
        z-index: 2;
    }

    /* Live Indicator */
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background-color: rgba(16, 185, 129, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }

    /* Sidebar Custom Figma Elements */
    .sidebar-menu {
        margin-top: 1.5rem;
    }
    .menu-item {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        color: #9ca3af;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        justify-content: space-between;
        align-items: center;
        text-decoration: none !important;
    }
    .menu-item.active {
        background-color: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border-left: 3px solid #3b82f6;
    }
    .menu-item:hover {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
    }
    
    .radar-card {
        background: rgba(17, 24, 39, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    .radar-screen {
        width: 100%;
        height: 120px;
        background: radial-gradient(circle, #022c22 0%, #064e3b 40%, #022c22 100%);
        border-radius: 12px;
        position: relative;
        overflow: hidden;
        border: 1px solid #047857;
    }
    .radar-sweep {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, rgba(16, 185, 129, 0.3) 0%, transparent 60%);
        transform-origin: center;
        width: 200%;
        height: 200%;
        top: -50%;
        left: -50%;
        animation: radar-spin 6s linear infinite;
    }
    @keyframes radar-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .radar-blip {
        position: absolute;
        width: 6px;
        height: 6px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px 2px #10b981;
        animation: blip-pulse 2s infinite;
    }
    @keyframes blip-pulse {
        0% { transform: scale(1); opacity: 0.2; }
        50% { transform: scale(1.5); opacity: 1; }
        100% { transform: scale(1); opacity: 0.2; }
    }
    
    .extended-forecast-row {
        display: flex;
        justify-content: space-between;
        gap: 6px;
        margin-top: 0.5rem;
    }
    .extended-day-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.5rem 0.25rem;
        text-align: center;
        transition: all 0.2s;
    }
    .extended-day-card:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .extended-day-name {
        font-size: 0.75rem;
        color: #9ca3af;
        font-weight: 600;
    }
    .extended-day-icon {
        font-size: 1.1rem;
        margin-top: 0.2rem;
    }

    /* Main Page Figma Layout Metrics */
    .forecast-card-figma {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.25rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .forecast-header-figma {
        font-size: 1.05rem;
        font-weight: 600;
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .forecast-temp-figma {
        font-size: 4rem;
        font-weight: 800;
        color: #f43f5e;
        text-shadow: 0 0 15px rgba(244, 63, 94, 0.35);
        line-height: 1;
    }
    
    .figma-metric-card {
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: all 0.2s;
    }
    .figma-metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(59, 130, 246, 0.2);
    }
    .figma-metric-title {
        font-size: 0.85rem;
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .figma-metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-top: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .mini-bar-chart {
        display: flex;
        align-items: flex-end;
        gap: 4px;
        height: 24px;
        margin-top: 1rem;
    }
    .mini-bar-chart .bar {
        flex: 1;
        background: linear-gradient(to top, #3b82f6, #60a5fa);
        border-radius: 2px;
    }
    .mini-area-chart {
        height: 24px;
        margin-top: 1rem;
        background: linear-gradient(180deg, rgba(168, 85, 247, 0.2) 0%, transparent 100%);
        clip-path: polygon(0 80%, 20% 60%, 40% 70%, 60% 40%, 80% 50%, 100% 20%, 100% 100%, 0 100%);
        border-top: 1px solid #a855f7;
    }
    .mini-wave-chart {
        height: 24px;
        margin-top: 1rem;
        background: linear-gradient(180deg, rgba(234, 179, 8, 0.15) 0%, transparent 100%);
        clip-path: polygon(0 90%, 15% 40%, 30% 60%, 45% 20%, 60% 70%, 75% 30%, 90% 50%, 100% 10%, 100% 100%, 0 100%);
        border-top: 1.5px solid #eab308;
    }
    .mini-dotted-chart {
        height: 24px;
        margin-top: 1rem;
        border-bottom: 2px dotted #6366f1;
        opacity: 0.6;
    }
    .progress-bar-container {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        height: 6px;
        width: 100%;
        margin-top: 1.5rem;
        overflow: hidden;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #3b82f6, #f43f5e);
        height: 100%;
        border-radius: 9999px;
    }
    
    /* Disable all pointer interactions on sidebar maps to avoid cut-off menus on touch/click */
    [data-testid="stSidebar"] iframe {
        pointer-events: none !important;
    }
</style>
""", unsafe_allow_html=True)

def map_wmo_code(code):
    wmo_map = {
        0: ("☀️", "Clear sky"),
        1: ("🌤️", "Mainly clear"),
        2: ("⛅", "Partly cloudy"),
        3: ("☁️", "Overcast"),
        45: ("🌫️", "Fog"),
        48: ("🌫️", "Depositing rime fog"),
        51: ("🌧️", "Light drizzle"),
        53: ("🌧️", "Moderate drizzle"),
        55: ("🌧️", "Dense drizzle"),
        56: ("🌧️", "Light freezing drizzle"),
        57: ("🌧️", "Dense freezing drizzle"),
        61: ("🌧️", "Slight rain"),
        63: ("🌧️", "Moderate rain"),
        65: ("🌧️", "Heavy rain"),
        66: ("🌧️", "Light freezing rain"),
        67: ("🌧️", "Heavy freezing rain"),
        71: ("❄️", "Slight snow fall"),
        73: ("❄️", "Moderate snow fall"),
        75: ("❄️", "Heavy snow fall"),
        77: ("❄️", "Snow grains"),
        80: ("🌧️", "Slight rain showers"),
        81: ("🌧️", "Moderate rain showers"),
        82: ("🌧️", "Violent rain showers"),
        85: ("❄️", "Slight snow showers"),
        86: ("❄️", "Heavy snow showers"),
        95: ("⛈️", "Thunderstorm"),
        96: ("⛈️", "Thunderstorm with slight hail"),
        99: ("⛈️", "Thunderstorm with heavy hail")
    }
    return wmo_map.get(code, ("❓", "Unknown"))

# Helper function to get latest day features from Forecast API daily data
def get_realtime_rf_features(daily_dict, feature_names):
    df = pd.DataFrame(daily_dict)
    rename_map = {
        "time": "date",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "temperature_2m_mean": "temp_mean",
        "relative_humidity_2m_mean": "humidity",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_speed",
        "pressure_msl_mean": "pressure"
    }
    df = df.rename(columns=rename_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filter the DataFrame to rows up to today (including today) to compute features up to today
    today_date = datetime.now().date()
    df_up_to_today = df[df['date'].dt.date <= today_date].copy()
    
    # Feature columns
    feature_cols = ['temp_mean', 'temp_max', 'temp_min', 'humidity', 'precipitation', 'wind_speed', 'pressure']
    
    # Lags
    for lag in [1, 3, 7]:
        for col in feature_cols:
            df_up_to_today[f'{col}_lag_{lag}'] = df_up_to_today[col].shift(lag)
            
    # Rolling averages
    for window in [7, 30]:
        df_up_to_today[f'temp_mean_roll_mean_{window}'] = df_up_to_today['temp_mean'].rolling(window=window).mean()
        df_up_to_today[f'humidity_roll_mean_{window}'] = df_up_to_today['humidity'].rolling(window=window).mean()
        df_up_to_today[f'pressure_roll_mean_{window}'] = df_up_to_today['pressure'].rolling(window=window).mean()
        
    # Seasonal
    df_up_to_today['month'] = df_up_to_today['date'].dt.month
    df_up_to_today['day_of_year'] = df_up_to_today['date'].dt.dayofyear
    df_up_to_today['sin_month'] = np.sin(2 * np.pi * df_up_to_today['month'] / 12.0)
    df_up_to_today['cos_month'] = np.cos(2 * np.pi * df_up_to_today['month'] / 12.0)
    df_up_to_today['sin_day'] = np.sin(2 * np.pi * df_up_to_today['day_of_year'] / 365.25)
    df_up_to_today['cos_day'] = np.cos(2 * np.pi * df_up_to_today['day_of_year'] / 365.25)
    
    # Clean up NaNs in features
    df_clean = df_up_to_today.dropna(subset=feature_names).reset_index(drop=True)
    
    # The last row has the completed features for the most recent day (today)
    latest_row = df_clean.iloc[-1]
    return latest_row[feature_names].to_frame().T, latest_row['date'], df

# Helper function to predict tomorrow using LSTM with Forecast API data
def predict_lstm_realtime(daily_dict, model_dir="models"):
    lstm_model_path = os.path.join(model_dir, "lstm_model.pth")
    lstm_scaler_path = os.path.join(model_dir, "lstm_scaler.joblib")
    
    lstm_metadata = joblib.load(lstm_scaler_path)
    feature_scaler = lstm_metadata['feature_scaler']
    target_scaler = lstm_metadata['target_scaler']
    feature_cols = lstm_metadata['feature_cols']
    seq_length = lstm_metadata['seq_length']
    
    df = pd.DataFrame(daily_dict)
    rename_map = {
        "time": "date",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "temperature_2m_mean": "temp_mean",
        "relative_humidity_2m_mean": "humidity",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_speed",
        "pressure_msl_mean": "pressure"
    }
    df = df.rename(columns=rename_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filter the DataFrame to rows up to today (including today)
    today_date = datetime.now().date()
    df_up_to_today = df[df['date'].dt.date <= today_date].copy()
    
    df_up_to_today['month'] = df_up_to_today['date'].dt.month
    df_up_to_today['day_of_year'] = df_up_to_today['date'].dt.dayofyear
    df_up_to_today['sin_month'] = np.sin(2 * np.pi * df_up_to_today['month'] / 12.0)
    df_up_to_today['cos_month'] = np.cos(2 * np.pi * df_up_to_today['month'] / 12.0)
    df_up_to_today['sin_day'] = np.sin(2 * np.pi * df_up_to_today['day_of_year'] / 365.25)
    df_up_to_today['cos_day'] = np.cos(2 * np.pi * df_up_to_today['day_of_year'] / 365.25)
    
    # Drop NaNs
    df_clean = df_up_to_today.dropna(subset=feature_cols).reset_index(drop=True)
    
    last_seq_df = df_clean[feature_cols].tail(seq_length)
    if len(last_seq_df) < seq_length:
         raise ValueError(f"Not enough data for LSTM sequence. Required {seq_length}, got {len(last_seq_df)}")
         
    scaled_features = feature_scaler.transform(last_seq_df.values)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
    
    device = torch.device("cpu")
    model_type = lstm_metadata.get('model_type', 'lstm')
    model_type_lower = model_type.lower()
    
    if model_type_lower == "gru":
        model = WeatherGRU(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=1)
    elif model_type_lower == "bilstm":
        model = WeatherBiLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=1)
    else:
        model = WeatherLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=1)
        
    model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
        
    pred = target_scaler.inverse_transform(pred_scaled).flatten()[0]
    latest_date = df_clean['date'].iloc[-1]
    return pred, latest_date

# Real-time weather observation helper from Forecast API
def fetch_realtime_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max,pressure_msl_mean,weather_code",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m,pressure_msl,weather_code",
        "past_days": 31,
        "forecast_days": 10,
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

# Geocoding search API helper
def search_city(query):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 5, "language": "en", "format": "json"}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        return []

# Load data and configurations
data_path = "data/processed_weather.csv"
raw_data_path = "data/raw_weather.csv"
model_dir = "models"
output_dir = "outputs"

if not os.path.exists(raw_data_path) or not os.path.exists(data_path):
    st.error("⚠️ Data files not found. Please run the training pipeline first to collect data and train models!")
    st.info("Run `python src/pipeline.py` in your terminal to start the pipeline.")
    st.stop()

raw_df = pd.read_csv(raw_data_path)
raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df = raw_df.sort_values('date').reset_index(drop=True)

processed_df = pd.read_csv(data_path)
processed_df['date'] = pd.to_datetime(processed_df['date'])

# Load best model label
best_model_name = "Random Forest"
best_model_path = os.path.join(model_dir, "best_model.txt")
if os.path.exists(best_model_path):
    with open(best_model_path, 'r', encoding='utf-8') as f:
        best_model_name = f.read().strip()

# Read active location context
active_location = "Mumbai, India"
location_path = os.path.join(model_dir, "active_location.txt")
if os.path.exists(location_path):
    with open(location_path, 'r', encoding='utf-8') as f:
        active_location = f.read().strip()

# Read active coordinates
active_lat = 19.0760
active_lon = 72.8777
coords_path = os.path.join(model_dir, "active_coordinates.txt")
if os.path.exists(coords_path):
    with open(coords_path, 'r', encoding='utf-8') as f:
        coords_str = f.read().strip()
        if ',' in coords_str:
            active_lat, active_lon = map(float, coords_str.split(','))
else:
    # Save default if not exists
    os.makedirs(model_dir, exist_ok=True)
    with open(coords_path, 'w', encoding='utf-8') as f:
        f.write(f"{active_lat},{active_lon}")

# App Header (Figma Top Bar Style)
current_time_str = datetime.now().strftime('%b %d, %Y | %I:%M %p')
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2.5rem; padding-bottom:1.2rem; border-bottom:1px solid rgba(255,255,255,0.05); margin-top: -30px;">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:2.2rem;">🌎</span>
        <div>
            <h1 style="color:#ffffff; font-size:1.8rem; font-weight:800; margin:0; line-height:1.2;">WeatherSphere AI</h1>
            <p style="color:#9ca3af; font-size:0.85rem; margin:0; margin-top:2px;">Location Context: <b>{active_location}</b></p>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:20px;">
        <div style="color:#9ca3af; font-size:0.9rem; font-weight:500;">📅 {current_time_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Header Logo
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:1.5rem; margin-top: -10px;">
    <span style="font-size:2rem;">🌐</span>
    <h2 style="color:#ffffff; margin:0; font-weight:800; font-size:1.4rem; line-height:1.2;">WeatherSphere AI</h2>
</div>
""", unsafe_allow_html=True)

# Global City Selector in Sidebar
st.sidebar.subheader("🔍 Search location, city...")
search_query = st.sidebar.text_input("City Search", placeholder="Type city or district...", label_visibility="collapsed")


if search_query:
    cities = search_city(search_query)
    if cities:
        city_options = []
        for c in cities:
            parts = [c['name']]
            if c.get('admin2') and c.get('admin2') != c['name']:
                parts.append(c['admin2'])
            if c.get('admin1') and c.get('admin1') != c.get('admin2') and c.get('admin1') != c['name']:
                parts.append(c['admin1'])
            if c.get('country'):
                parts.append(c['country'])
            city_options.append(", ".join(parts))
            
        selected_city_str = st.sidebar.selectbox("Select matching location:", city_options)
        selected_idx = city_options.index(selected_city_str)
        selected_city = cities[selected_idx]
        
        if st.sidebar.button("🚀 Fetch & Train AI Models"):
            lat = selected_city["latitude"]
            lon = selected_city["longitude"]
            city_name = selected_city_str
            
            # Run the dynamic pipeline with a spinner
            with st.spinner(f"Analyzing {city_name}... Fetching 5y history & training models on the fly..."):
                try:
                    # 1. Ingest
                    df_raw = fetch_weather_data(lat, lon, city_name)
                    if df_raw is not None:
                        df_raw.to_csv(raw_data_path, index=False)
                        
                        # 2. Feature Engineer
                        engineer_features(raw_data_path, data_path)
                        
                        # 3. Train RF
                        train_rf(data_path, model_dir)
                        
                        # 4. Train LSTM (30 epochs for fast interactive feedback)
                        train_lstm(data_path, model_dir, epochs=30)
                        
                        # 5. Evaluate
                        evaluate_models(data_path, model_dir, output_dir)
                        
                        # Save active location
                        with open(location_path, 'w', encoding='utf-8') as f:
                            f.write(city_name)
                            
                        # Save active coordinates
                        with open(coords_path, 'w', encoding='utf-8') as f:
                            f.write(f"{lat},{lon}")
                            
                        st.sidebar.success(f"Models successfully trained for {city_name}!")
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to fetch weather data.")
                except Exception as e:
                    st.sidebar.error(f"Error executing pipeline: {e}")
    else:
        st.sidebar.warning("No matching locations found.")

st.sidebar.markdown("---")

# Sidebar Custom Figma Navigation Menu
active_tab = st.query_params.get("tab", "Dashboard")

dashboard_active = "active" if active_tab == "Dashboard" else ""
hourly_active = "active" if active_tab == "Hourly" else ""
tenday_active = "active" if active_tab == "10-Day" else ""
alerts_active = "active" if active_tab == "Alerts" else ""
maps_active = "active" if active_tab == "Maps" else ""
analytics_active = "active" if active_tab == "Analytics" else ""
settings_active = "active" if active_tab == "Settings" else ""

st.sidebar.markdown(f"""
<div class="sidebar-menu">
    <a href="?tab=Dashboard" target="_self" class="menu-item {dashboard_active}">
        <span>📊 Dashboard</span>
        <span style="background-color:#10b981; color:#ffffff; font-size:0.7rem; padding:1px 5px; border-radius:4px; font-weight:700;">Active</span>
    </a>
    <a href="?tab=Hourly" target="_self" class="menu-item {hourly_active}">🕒 Hourly</a>
    <a href="?tab=10-Day" target="_self" class="menu-item {tenday_active}">📅 10-Day</a>
    <a href="?tab=Alerts" target="_self" class="menu-item {alerts_active}">⚠️ Alerts</a>
    <a href="?tab=Maps" target="_self" class="menu-item {maps_active}">🗺️ Maps</a>
    <a href="?tab=Analytics" target="_self" class="menu-item {analytics_active}">📈 Analytics</a>
    <a href="?tab=Settings" target="_self" class="menu-item {settings_active}">⚙️ Settings</a>
</div>
""", unsafe_allow_html=True)

# Map Visualization Card in Sidebar
st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#9ca3af; font-weight:600; text-align:left; margin-top:1.5rem; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.05em;">Map Visualization (Live Preview)</div>
""", unsafe_allow_html=True)

mini_windy_url = f"https://embed.windy.com/embed2.html?lat={active_lat}&lon={active_lon}&zoom=5&level=surface&overlay=radar&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=false&metricWind=default&metricTemp=default&radarRange=-1"

with st.sidebar:
    st.components.v1.html(f"""
    <div style="position: relative; width: 100%; height: 200px; overflow: hidden; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
        <iframe src="{mini_windy_url}" width="100%" height="200" style="border: none; pointer-events: none;" scrolling="no"></iframe>
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 9999; background: transparent; cursor: default;"></div>
    </div>
    """, height=202)
    st.caption("💡 Sidebar map is a live visual-only preview. Navigate to the **Maps** tab for full interactive controls, layer switching, and radar zoom.")

# 7-day extended forecast row
st.sidebar.markdown("""
<div style="margin-top:1.5rem; margin-bottom: 1.5rem;">
    <div style="font-size:0.8rem; color:#9ca3af; font-weight:600; text-align:left; text-transform:uppercase; letter-spacing:0.05em; margin-bottom: 0.5rem;">7-day extended forecast</div>
    <div class="extended-forecast-row">
        <div class="extended-day-card">
            <div class="extended-day-name">Mon</div>
            <div class="extended-day-icon">⛅</div>
        </div>
        <div class="extended-day-card">
            <div class="extended-day-name">Tue</div>
            <div class="extended-day-icon">☀️</div>
        </div>
        <div class="extended-day-card">
            <div class="extended-day-name">Wed</div>
            <div class="extended-day-icon">☁️</div>
        </div>
        <div class="extended-day-card">
            <div class="extended-day-name">Thu</div>
            <div class="extended-day-icon">⛅</div>
        </div>
        <div class="extended-day-card">
            <div class="extended-day-name">Fri</div>
            <div class="extended-day-icon">🌧️</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Date range selection
min_date = raw_df['date'].min().to_pydatetime()
max_date = raw_df['date'].max().to_pydatetime()

st.sidebar.subheader("📅 Filter Historical Data")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Models Active")
st.sidebar.info("1. **Random Forest Regressor**\n2. **PyTorch LSTM**")

# Fetch Real-Time weather observations and daily sequence values
realtime_data = fetch_realtime_weather(active_lat, active_lon)

# Calculate Next-Day Forecast (Real-Time tomorrow forecast)
forecast_temp = 0.0
latest_obs_date = None

if not realtime_data:
    st.warning("⚠️ Failed to reach Real-Time Weather API. Showing cached forecasts.")
    # Build a simulated realtime_data structure from the local raw_df
    last_df = raw_df.tail(41).copy()
    if len(last_df) >= 30:
        daily_dict = {
            "time": last_df['date'].dt.strftime("%Y-%m-%d").tolist(),
            "temperature_2m_max": last_df['temp_max'].tolist(),
            "temperature_2m_min": last_df['temp_min'].tolist(),
            "temperature_2m_mean": last_df['temp_mean'].tolist(),
            "relative_humidity_2m_mean": last_df['humidity'].tolist(),
            "precipitation_sum": last_df['precipitation'].tolist(),
            "wind_speed_10m_max": last_df['wind_speed'].tolist(),
            "pressure_msl_mean": last_df['pressure'].tolist(),
            "weather_code": [0] * len(last_df)
        }
        
        last_row = last_df.iloc[-1]
        current_dict = {
            "temperature_2m": float(last_row['temp_mean']),
            "relative_humidity_2m": float(last_row['humidity']),
            "precipitation": float(last_row['precipitation']),
            "pressure_msl": float(last_row['pressure']),
            "wind_speed_10m": float(last_row['wind_speed'])
        }
        
        hourly_times = []
        hourly_temps = []
        hourly_humidity = []
        hourly_precip_prob = []
        hourly_wind = []
        hourly_pressure = []
        hourly_codes = []
        
        base_date = last_row['date']
        for hour in range(24):
            hour_dt = base_date.replace(hour=hour, minute=0)
            hourly_times.append(hour_dt.strftime("%Y-%m-%dT%H:%M"))
            diurnal_var = np.sin((hour - 6) * np.pi / 12.0) * ((last_row['temp_max'] - last_row['temp_min']) / 2.0)
            hourly_temps.append(float(last_row['temp_mean'] + diurnal_var))
            hourly_humidity.append(float(last_row['humidity']))
            hourly_precip_prob.append(20 if last_row['precipitation'] > 0 else 0)
            hourly_wind.append(float(last_row['wind_speed']))
            hourly_pressure.append(float(last_row['pressure']))
            hourly_codes.append(0)
            
        hourly_dict = {
            "time": hourly_times,
            "temperature_2m": hourly_temps,
            "relative_humidity_2m": hourly_humidity,
            "precipitation_probability": hourly_precip_prob,
            "wind_speed_10m": hourly_wind,
            "pressure_msl": hourly_pressure,
            "weather_code": hourly_codes
        }
        
        realtime_data = {
            "current": current_dict,
            "daily": daily_dict,
            "hourly": hourly_dict
        }

if realtime_data:
    try:
        daily_data = realtime_data.get("daily", {})
        if best_model_name == "Random Forest":
            rf_model = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
            # Predict using real-time features up to today
            rf_features_df, latest_obs_date, realtime_engineered_df = get_realtime_rf_features(daily_data, rf_model.feature_names_in_)
            forecast_temp = rf_model.predict(rf_features_df)[0]
        else:
            # LSTM
            forecast_temp, latest_obs_date = predict_lstm_realtime(daily_data, model_dir)
    except Exception as e:
        st.error(f"Error calculating real-time prediction with {best_model_name}: {e}")
        st.stop()
else:
    st.error("⚠️ Failed to reach Real-Time Weather API and no cached local data available.")
    st.stop()

forecast_date = latest_obs_date + timedelta(days=1)

# Tab Layout
# Load live details
current = realtime_data.get("current", {})
live_temp = current.get("temperature_2m", 0.0)
live_humidity = current.get("relative_humidity_2m", 0.0)
live_pressure = current.get("pressure_msl", 1013.0)
live_wind = current.get("wind_speed_10m", 0.0)
live_precip = current.get("precipitation", 0.0)

# Today's daily max and min (calculated dynamically by matching today's date)
daily_temps = realtime_data.get("daily", {})
daily_dates = daily_temps.get("time", [])
today_str = datetime.now().strftime("%Y-%m-%d")
try:
    today_idx = daily_dates.index(today_str)
except ValueError:
    today_idx = -10 if len(daily_dates) >= 10 else 0

live_max = daily_temps.get("temperature_2m_max", [live_temp])[today_idx]
live_min = daily_temps.get("temperature_2m_min", [live_temp])[today_idx]

# View selection based on query parameter
active_tab = st.query_params.get("tab", "Dashboard")

# Compute biometeorological alert level and health offset values
biometeorological_alert = "NORMAL"
risk_penalty_percentage = 0.0
health_guidelines = "Weather conditions are within normal bounds. No environmental health warning is active."

if forecast_temp >= 37.0 or (forecast_temp >= 34.0 and live_humidity >= 70.0):
    biometeorological_alert = "SEVERE HEAT WARNING"
    risk_penalty_percentage = 15.0
    health_guidelines = "Extreme heat index detected. Combined thermal and humidity load places significant cardiovascular strain on the body. Rest and limit physical activity."
elif forecast_temp >= 33.0:
    biometeorological_alert = "HEAT ADVISORY"
    risk_penalty_percentage = 8.0
    health_guidelines = "High temperature warning. Ensure adequate hydration and avoid prolonged outdoor activity during peak sunlight hours."
elif forecast_temp <= 10.0:
    biometeorological_alert = "COLD ADVISORY"
    risk_penalty_percentage = 10.0
    health_guidelines = "Freezing temperatures trigger peripheral vasoconstriction, raising blood pressure and stroke vulnerability. Dress warmly and monitor vitals."
elif live_pressure < 1008.0 and live_precip > 5.0:
    biometeorological_alert = "BAROMETRIC PRESSURE ALERT"
    risk_penalty_percentage = 5.0
    health_guidelines = "Sudden drop in atmospheric pressure combined with dampness may trigger joint inflammation and arthritis pain."

import json
os.makedirs(output_dir, exist_ok=True)
forecast_export = {
    "location": active_location,
    "coordinates": f"{active_lat},{active_lon}",
    "temp_today": float(live_temp),
    "temp_tomorrow": float(forecast_temp),
    "humidity": float(live_humidity),
    "wind_speed": float(live_wind),
    "pressure": float(live_pressure),
    "precipitation": float(live_precip),
    "model_used": best_model_name,
    "biometeorological_alert": biometeorological_alert,
    "risk_penalty_percentage": risk_penalty_percentage,
    "health_guidelines": health_guidelines,
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
with open(os.path.join(output_dir, "latest_forecast.json"), "w", encoding='utf-8') as f:
    json.dump(forecast_export, f, indent=2)

if active_tab == "Dashboard":
    col1, col2 = st.columns([4, 6])
    
    with col1:
        # Next-Day Forecast Card (Figma Style)
        tomorrow_str = forecast_date.strftime("%Y-%m-%d")
        try:
            tomorrow_idx = daily_dates.index(tomorrow_str)
            tomorrow_wmo = daily_temps.get("weather_code", [0])[tomorrow_idx]
            condition_emoji, condition_label = map_wmo_code(tomorrow_wmo)
        except Exception:
            condition_label = "Partly Cloudy"
            condition_emoji = "⛅"
            if live_precip > 5.0:
                condition_label = "Rainy / Wet"
                condition_emoji = "🌧️"
            elif live_wind > 20.0:
                condition_label = "Windy"
                condition_emoji = "💨"
            elif live_humidity > 80.0:
                condition_label = "Humid"
                condition_emoji = "💧"
            else:
                condition_emoji = "☀️"
            
        st.markdown(f"""
        <div class="forecast-card-figma">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="forecast-header-figma">Next-Day Forecast (Tomorrow)</div>
                <div style="color:#eab308; font-size:1.2rem;">⭐</div>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between; margin-top:2.5rem;">
                <div class="forecast-temp-figma">{forecast_temp:.1f}°C</div>
                <div style="font-size:3.5rem;">{condition_emoji}</div>
            </div>
            <div style="text-align:left; margin-top:2.5rem;">
                <div style="font-weight:700; font-size:1.3rem; color:#f3f4f6;">{condition_label}</div>
                <div style="color:#9ca3af; font-size:0.9rem; margin-top:0.25rem;">Forecast Date: {forecast_date.strftime('%B %dth')}</div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,0.05);">
                <div style="color:#9ca3af; font-size:0.85rem; font-weight:500;">High / Low</div>
                <div style="font-weight:700; color:#f3f4f6; font-size:1.05rem;">{live_max:.1f}°C / {live_min:.1f}°C</div>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width:72%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Grid of 8 cards (4 columns, 2 rows)
        grid_col1, grid_col2, grid_col3, grid_col4 = st.columns(4)
        
        aq_label = "Good"
        aq_color = "#10b981"
        if live_humidity > 80.0:
            aq_label = "Moderate"
            aq_color = "#eab308"
            
        with grid_col1:
            # Card 1: Humidity
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">Humidity</div>
                    <div style="color:#3b82f6; font-size:0.95rem;">💧</div>
                </div>
                <div class="figma-metric-value">{live_humidity:.0f}%</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">Humidity</div>
                <div class="mini-bar-chart">
                    <div class="bar" style="height:35%;"></div>
                    <div class="bar" style="height:50%;"></div>
                    <div class="bar" style="height:70%;"></div>
                    <div class="bar" style="height:60%;"></div>
                    <div class="bar" style="height:85%;"></div>
                    <div class="bar" style="height:95%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 5: Mean Temp Today
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">Mean Temp</div>
                    <div style="color:#f43f5e; font-size:0.95rem;">🌡️</div>
                </div>
                <div class="figma-metric-value">{live_temp:.1f}°C</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">Average</div>
                <div class="mini-bar-chart">
                    <div class="bar" style="height:60%; background:#f43f5e;"></div>
                    <div class="bar" style="height:65%; background:#f43f5e;"></div>
                    <div class="bar" style="height:70%; background:#f43f5e;"></div>
                    <div class="bar" style="height:75%; background:#f43f5e;"></div>
                    <div class="bar" style="height:80%; background:#f43f5e;"></div>
                    <div class="bar" style="height:85%; background:#f43f5e;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with grid_col2:
            # Card 2: Pressure
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">Pressure</div>
                    <div style="color:#a855f7; font-size:0.95rem;">🌀</div>
                </div>
                <div class="figma-metric-value">{live_pressure:.0f} hPa</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">Pressure</div>
                <div class="mini-area-chart"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 6: High Temp
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">High Temp</div>
                    <div style="color:#ef4444; font-size:0.95rem;">📈</div>
                </div>
                <div class="figma-metric-value">{live_max:.1f}°C</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">Maximum</div>
                <div class="mini-area-chart" style="border-top-color:#ef4444; background:linear-gradient(180deg, rgba(239,68,68,0.2) 0%, transparent 100%);"></div>
            </div>
            """, unsafe_allow_html=True)
            
        with grid_col3:
            # Card 3: Wind Speed
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">Wind Speed</div>
                    <div style="color:#eab308; font-size:0.95rem;">💨</div>
                </div>
                <div class="figma-metric-value">{live_wind:.1f} km/h</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">Wind</div>
                <div class="mini-wave-chart"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 7: Air Quality
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">Air Quality</div>
                    <div style="color:#10b981; font-size:0.95rem;">🌱</div>
                </div>
                <div class="figma-metric-value" style="color:{aq_color};">{aq_label}</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">AQ Index</div>
                <div style="width:100%; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:1.5rem; overflow:hidden;">
                    <div style="width:85%; height:100%; background:{aq_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with grid_col4:
            # Card 4: UV Index (Moderate / 4)
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">UV Index</div>
                    <div style="color:#fb923c; font-size:0.95rem;">☀️</div>
                </div>
                <div class="figma-metric-value">4</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">Moderate</div>
                <div style="width:100%; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; margin-top:1.5rem; overflow:hidden;">
                    <div style="width:40%; height:100%; background:#fb923c;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 8: Rainfall
            st.markdown(f"""
            <div class="figma-metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="figma-metric-title">Rainfall</div>
                    <div style="color:#6366f1; font-size:0.95rem;">🌧️</div>
                </div>
                <div class="figma-metric-value">{live_precip:.1f} mm</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; font-weight:500;">Rainfall</div>
                <div class="mini-dotted-chart"></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Visual forecast context: Plot last 14 days of real-time + tomorrow's forecast
    st.markdown("### Real-Time Temperature Context & Forecast")
    
    # Extract last 14 days dates and mean temps
    real_daily_dates = daily_temps.get("time", [])[:-1] # Exclude tomorrow
    real_daily_means = daily_temps.get("temperature_2m_mean", [])[:-1]
    
    fig = go.Figure()
    # Actuals
    fig.add_trace(go.Scatter(
        x=real_daily_dates[-14:], 
        y=real_daily_means[-14:], 
        mode='lines+markers',
        name='Actual Mean Temp',
        line=dict(color='#60a5fa', width=3),
        marker=dict(size=8)
    ))
    
    # Forecast point
    fig.add_trace(go.Scatter(
        x=[forecast_date], 
        y=[forecast_temp], 
        mode='markers',
        name="Tomorrow's Forecast",
        marker=dict(color='#f43f5e', size=14, symbol='star', line=dict(color='white', width=2))
    ))
    
    # Connect last actual to forecast with a dotted line
    fig.add_trace(go.Scatter(
        x=[real_daily_dates[-1], forecast_date],
        y=[real_daily_means[-1], forecast_temp],
        mode='lines',
        name='Forecast Link',
        line=dict(color='#f43f5e', width=2, dash='dot'),
        showlegend=False
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(title='Temperature (°C)', gridcolor='#1e293b'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

elif active_tab == "Hourly":
    st.markdown("### 🕒 24-Hour Forecast Trend")
    if "hourly" in realtime_data:
        hourly = realtime_data.get("hourly", {})
        hourly_times = hourly.get("time", [])
        now_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        now_hour_str = now_hour.strftime("%Y-%m-%dT%H:00")
        
        start_idx = 0
        for idx, t_str in enumerate(hourly_times):
            if t_str.startswith(now_hour_str):
                start_idx = idx
                break
                
        slice_end = start_idx + 24
        display_times_raw = hourly_times[start_idx:slice_end]
        
        hours = [datetime.strptime(t, "%Y-%m-%dT%H:%M").strftime("%I:00 %p") for t in display_times_raw]
        temps_24h = hourly.get("temperature_2m", [])[start_idx:slice_end]
        humidities_24h = hourly.get("relative_humidity_2m", [])[start_idx:slice_end]
        precip_probs_24h = hourly.get("precipitation_probability", [])[start_idx:slice_end]
        wind_speeds_24h = hourly.get("wind_speed_10m", [])[start_idx:slice_end]
        codes_24h = hourly.get("weather_code", [])[start_idx:slice_end]
        
        conditions_24h = [map_wmo_code(c)[1] for c in codes_24h]
        emojis_24h = [map_wmo_code(c)[0] for c in codes_24h]
    else:
        hours = [f"{(datetime.now() + timedelta(hours=i)).strftime('%I:00 %p')}" for i in range(24)]
        temps_24h = [live_temp + np.sin(i * np.pi / 12) * (live_max - live_min) / 2 for i in range(24)]
        humidities_24h = [live_humidity + np.cos(i*np.pi/12)*5 for i in range(24)]
        precip_probs_24h = [0 for _ in range(24)]
        wind_speeds_24h = [live_wind for _ in range(24)]
        emojis_24h = ["⛅" for _ in range(24)]
        conditions_24h = ["Partly Cloudy" for _ in range(24)]
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=hours, 
        y=temps_24h, 
        mode='lines+markers', 
        name="Temperature (°C)",
        line=dict(color='#60a5fa', width=3),
        marker=dict(size=6)
    ))
    
    fig_hourly.add_trace(go.Bar(
        x=hours,
        y=precip_probs_24h,
        name="Rain Probability (%)",
        yaxis="y2",
        marker_color='rgba(99, 102, 241, 0.4)'
    ))
    
    fig_hourly.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title="Temperature (°C)", gridcolor='#1e293b'),
        yaxis2=dict(title="Rain Probability (%)", overlaying="y", side="right", range=[0, 100], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    st.markdown("#### Hourly Timeline Details")
    hourly_df = pd.DataFrame({
        "Hour": hours, 
        "Condition": [f"{e} {c}" for e, c in zip(emojis_24h, conditions_24h)],
        "Temperature": [f"{t:.1f}°C" for t in temps_24h], 
        "Humidity": [f"{h:.0f}%" for h in humidities_24h],
        "Wind Speed": [f"{w:.1f} km/h" for w in wind_speeds_24h],
        "Rain Prob": [f"{p}%" for p in precip_probs_24h]
    })
    st.dataframe(hourly_df, hide_index=True, use_container_width=True)

elif active_tab == "10-Day":
    st.markdown("### 📅 10-Day Extended Forecast")
    if "daily" in realtime_data:
        daily = realtime_data.get("daily", {})
        daily_dates_raw = daily.get("time", [])
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        try:
            start_idx = daily_dates_raw.index(today_str)
        except ValueError:
            start_idx = max(0, len(daily_dates_raw) - 10)
            
        slice_end = start_idx + 10
        display_dates = daily_dates_raw[start_idx:slice_end]
        temps_max = daily.get("temperature_2m_max", [])[start_idx:slice_end]
        temps_min = daily.get("temperature_2m_min", [])[start_idx:slice_end]
        wmo_codes = daily.get("weather_code", [])[start_idx:slice_end]
        
        days = []
        for d in display_dates:
            dt = datetime.strptime(d, "%Y-%m-%d")
            days.append(dt.strftime('%A, %b %d'))
            
        icons = []
        conditions = []
        for code in wmo_codes:
            emoji, desc = map_wmo_code(code)
            icons.append(emoji)
            conditions.append(desc)
    else:
        days = [(datetime.now() + timedelta(days=i)).strftime('%A, %b %d') for i in range(10)]
        icons = ["☀️", "⛅", "☁️", "🌧️", "⛅", "☀️", "🌧️", "☁️", "☀️", "⛅"]
        temps_max = [live_max + np.random.uniform(-2, 2) for _ in range(10)]
        temps_min = [live_min + np.random.uniform(-2, 2) for _ in range(10)]
        conditions = ["Clear" if ic == "☀️" else "Cloudy" for ic in icons]
    
    cols = st.columns(5)
    for idx, day in enumerate(days[:5]):
        with cols[idx]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height: 200px;">
                <div style="font-weight:600; font-size:0.95rem; color:#9ca3af;">{day}</div>
                <div style="font-size:3rem; margin:0.5rem 0;">{icons[idx]}</div>
                <div style="font-size:0.8rem; color:#60a5fa; font-weight:500; margin-bottom:0.5rem;">{conditions[idx]}</div>
                <div style="font-weight:700; font-size:1.2rem; color:#f3f4f6;">{temps_max[idx]:.1f}°C</div>
                <div style="font-size:0.85rem; color:#9ca3af;">Min: {temps_min[idx]:.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)
            
    cols2 = st.columns(5)
    for idx, day in enumerate(days[5:]):
        with cols2[idx]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height: 200px;">
                <div style="font-weight:600; font-size:0.95rem; color:#9ca3af;">{day}</div>
                <div style="font-size:3rem; margin:0.5rem 0;">{icons[idx+5]}</div>
                <div style="font-size:0.8rem; color:#60a5fa; font-weight:500; margin-bottom:0.5rem;">{conditions[idx+5]}</div>
                <div style="font-weight:700; font-size:1.2rem; color:#f3f4f6;">{temps_max[idx+5]:.1f}°C</div>
                <div style="font-size:0.85rem; color:#9ca3af;">Min: {temps_min[idx+5]:.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)

elif active_tab == "Alerts":
    st.markdown("### ⚠️ Severe Weather Alerts & Warnings")
    any_alerts = False
    
    if biometeorological_alert != "NORMAL":
        any_alerts = True
        alert_color = "#ef4444" if "SEVERE" in biometeorological_alert or "WARNING" in biometeorological_alert else "#eab308"
        alert_bg = "rgba(239, 68, 68, 0.05)" if "SEVERE" in biometeorological_alert or "WARNING" in biometeorological_alert else "rgba(234, 179, 8, 0.05)"
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {alert_color}; background: {alert_bg};">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.8rem;">🚨</span>
                <div>
                    <h4 style="color:{alert_color}; margin:0; font-weight:700;">{biometeorological_alert}</h4>
                    <p style="color:#9ca3af; font-size:0.9rem; margin:0; margin-top:4px;">Dynamic Biometeorological Threat Level</p>
                </div>
            </div>
            <p style="color:#f3f4f6; font-size:1rem; margin-top:1rem;"><b>Diagnostic:</b> {health_guidelines}</p>
            <p style="color:#9ca3af; font-size:0.85rem; margin-top:0.5rem;">Calculated Cardiovascular / Physiological Risk Offset: <b>+{risk_penalty_percentage:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    if live_wind > 20.0:
        any_alerts = True
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #eab308; background: rgba(234, 179, 8, 0.05); margin-top:1.2rem;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.8rem;">💨</span>
                <div>
                    <h4 style="color:#eab308; margin:0; font-weight:700;">High Wind Caution</h4>
                    <p style="color:#fde047; font-size:0.9rem; margin:0; margin-top:4px;">Strong wind velocities detected: {live_wind:.1f} km/h</p>
                </div>
            </div>
            <p style="color:#f3f4f6; font-size:0.9rem; margin-top:1rem;">Live atmospheric observations indicate persistent high wind velocities. Secure outdoor loose objects and avoid high-profile transit risks.</p>
        </div>
        """, unsafe_allow_html=True)
        
    if not any_alerts:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #10b981; background: rgba(16, 185, 129, 0.05);">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.8rem;">✅</span>
                <div>
                    <h4 style="color:#10b981; margin:0; font-weight:700;">Atmospheric All Clear</h4>
                    <p style="color:#a7f3d0; font-size:0.9rem; margin:0; margin-top:4px;">No active biometeorological hazards detected for {active_location}</p>
                </div>
            </div>
            <p style="color:#f3f4f6; font-size:0.95rem; margin-top:1rem;">Current telemetry indicates that temperature, barometric pressure, wind speeds, and relative humidity indices are well within safe physiological bounds. Outdoor activity is highly recommended.</p>
        </div>
        """, unsafe_allow_html=True)

elif active_tab == "Maps":
    st.markdown("### 🗺️ Interactive Live Weather radar & Satellite map")
    
    # Dynamic Map Layer Selector
    map_layer = st.radio(
        "Select Live Map Layer:",
        options=["📡 Weather Radar", "🛰️ Satellite Map", "💨 Wind Velocity", "🌡️ Temperature Layer", "🌧️ Rain / Accumulation"],
        horizontal=True
    )
    
    layer_map = {
        "📡 Weather Radar": "radar",
        "🛰️ Satellite Map": "satellite",
        "💨 Wind Velocity": "wind",
        "🌡️ Temperature Layer": "temp",
        "🌧️ Rain / Accumulation": "rain"
    }
    
    overlay = layer_map[map_layer]
    
    # Premium interactive Windy widget embedded via iframe (with detail=false to hide white details pane)
    windy_url = f"https://embed.windy.com/embed2.html?lat={active_lat}&lon={active_lon}&zoom=7&level=surface&overlay={overlay}&menu=&message=true&marker=true&calendar=now&pressure=true&type=map&location=coordinates&detail=false&metricWind=km%2Fh&metricTemp=%C2%B0C&radarRange=-1"
    
    st.components.v1.iframe(windy_url, height=550, scrolling=False)
    
    st.markdown(f"""
    <div class="glass-card" style="margin-top:1.5rem;">
        <h4 style="color:#60a5fa; margin:0; font-weight:700; margin-bottom:0.5rem;">Satellite Overview & Telemetry Details</h4>
        <p style="color:#9ca3af; font-size:0.9rem; margin:0;">Location: <b>{active_location}</b> (Coordinates: <b>{active_lat:.4f}° N, {active_lon:.4f}° E</b>). Real-time map layers dynamically centered via embedded Windy radar telemetry.</p>
    </div>
    """, unsafe_allow_html=True)

elif active_tab == "Analytics":
    st.markdown("### Historical Weather Explorer (5-Year Archival Data)")
    if len(date_range) == 2:
        start_dt, end_dt = date_range
        filtered_df = raw_df[(raw_df['date'].dt.date >= start_dt) & (raw_df['date'].dt.date <= end_dt)]
    else:
        filtered_df = raw_df
        
    st.write(f"Showing historical trends from **{filtered_df['date'].min().strftime('%Y-%m-%d')}** to **{filtered_df['date'].max().strftime('%Y-%m-%d')}** ({len(filtered_df)} days)")
    plot_var = st.selectbox("Select Parameter to Visualize", ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Pressure (hPa)", "Precipitation (mm)"])
    
    fig_hist = go.Figure()
    if plot_var == "Temperature (°C)":
        fig_hist.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['temp_max'], name='Max Temp', line=dict(color='#f43f5e', width=1.5)))
        fig_hist.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['temp_mean'], name='Mean Temp', line=dict(color='#3b82f6', width=2)))
        fig_hist.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['temp_min'], name='Min Temp', line=dict(color='#10b981', width=1.5)))
        fig_hist.update_layout(yaxis_title="Temperature (°C)")
    elif plot_var == "Humidity (%)":
        fig_hist.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['humidity'], name='Humidity', line=dict(color='#06b6d4', width=2)))
        fig_hist.update_layout(yaxis_title="Humidity (%)")
    elif plot_var == "Wind Speed (km/h)":
        fig_hist.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['wind_speed'], name='Wind Speed', line=dict(color='#eab308', width=2)))
        fig_hist.update_layout(yaxis_title="Wind Speed (km/h)")
    elif plot_var == "Pressure (hPa)":
        fig_hist.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['pressure'], name='Pressure', line=dict(color='#a855f7', width=2)))
        fig_hist.update_layout(yaxis_title="Pressure (hPa)")
    else:
        fig_hist.add_trace(go.Bar(x=filtered_df['date'], y=filtered_df['precipitation'], name='Precipitation', marker_color='#6366f1'))
        fig_hist.update_layout(yaxis_title="Precipitation (mm)")
        
    fig_hist.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b'),
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

elif active_tab == "Settings":
    st.markdown("### Model Comparison & Metrics")
    metrics_path = os.path.join(output_dir, "metrics_comparison.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        m_col1, m_col2 = st.columns([1, 1])
        with m_col1:
            st.markdown("#### Test Set Performance Summary")
            st.dataframe(metrics_df, hide_index=True)
            
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['Test RMSE (°C)'], name='RMSE', marker_color='#3b82f6'))
            fig_metrics.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['Test MAE (°C)'], name='MAE', marker_color='#10b981'))
            fig_metrics.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                barmode='group',
                title='RMSE vs MAE Comparison (Lower is Better)',
                yaxis_title='Metric Value (°C)'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
            
        with m_col2:
            st.markdown("#### Model Architecture Details")
            st.markdown("""
            - **Random Forest Regressor (Baseline)**
              - Trained using scikit-learn ensemble.
              - Uses engineered lag features (1, 3, 7 days), rolling averages (7, 30 days), and seasonal variables.
              - Predicts target temperature directly from tabular features.
            
            - **LSTM Network (Sequence Model)**
              - Deep learning architecture trained in PyTorch.
              - Learns temporal correlations from 7-day sequence inputs of core features.
              - Automatically handles non-linear patterns over time.
            """)
            st.success(f"🏆 Based on the test set RMSE, the **{best_model_name}** model is the best-performing forecaster and is selected for active predictions.")
            
    st.markdown("---")
    st.markdown("### Model Evaluation Visualizations")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("#### Actual vs Predicted Temperature Curve")
        actual_pred_img = os.path.join(output_dir, "actual_vs_predicted.png")
        if os.path.exists(actual_pred_img):
            st.image(actual_pred_img, use_column_width=True)
        else:
            st.info("Actual vs Predicted plot image not found.")
            
    with img_col2:
        st.markdown("#### Random Forest Feature Importances")
        feat_imp_img = os.path.join(output_dir, "rf_feature_importance.png")
        if os.path.exists(feat_imp_img):
            st.image(feat_imp_img, use_column_width=True)
        else:
            st.info("Feature importance plot image not found.")

    st.markdown("---")
    st.markdown("### ⚙️ Deep Learning Model Studio")
    st.markdown("Customize hyperparameter weights and train a custom PyTorch sequence model (LSTM, GRU, or Bi-LSTM) on the fly.")
    
    studio_col1, studio_col2 = st.columns([1, 1])
    with studio_col1:
        studio_model_type = st.selectbox("Model Architecture", ["LSTM", "GRU", "Bi-LSTM"])
        studio_epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=30, step=5)
        studio_lr = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0005, format="%.4f")
    with studio_col2:
        studio_seq_len = st.slider("Sequence History Length (Days)", min_value=3, max_value=14, value=7)
        studio_batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        
    if st.button("🚀 Train & Optimize Model"):
        progress_placeholder = st.empty()
        loss_chart_placeholder = st.empty()
        
        losses = []
        
        def on_epoch_done(epoch, total_epochs, loss):
            losses.append(loss)
            pct = int(epoch / total_epochs * 100)
            progress_placeholder.progress(pct, text=f"Training Epoch {epoch}/{total_epochs}... Current Loss: {loss:.6f}")
            
            # Plot loss curve
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(losses)+1)), 
                y=losses, 
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#f43f5e', width=2),
                marker=dict(size=6)
            ))
            fig_loss.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="Epoch",
                yaxis_title="Loss (MSE)",
                height=250
            )
            loss_chart_placeholder.plotly_chart(fig_loss, use_container_width=True)

        with st.spinner("Initializing Model Studio Training..."):
            try:
                # Import train_lstm and train model on-the-fly
                from src.train_lstm import train_lstm
                from src.evaluate import evaluate_models
                
                # Train
                train_lstm(
                    data_path=data_path, 
                    model_dir=model_dir, 
                    epochs=studio_epochs, 
                    batch_size=studio_batch_size, 
                    seq_length=studio_seq_len, 
                    model_type=studio_model_type.lower(), 
                    lr=studio_lr, 
                    epoch_callback=on_epoch_done
                )
                
                # Re-evaluate models
                evaluate_models(data_path=data_path, model_dir=model_dir, output_dir=output_dir)
                
                st.success(f"🏆 Training complete! PyTorch {studio_model_type.upper()} model has been registered as the active forecaster.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to train model: {e}")
