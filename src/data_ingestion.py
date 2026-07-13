import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_data(lat=19.0760, lon=72.8777, city="Mumbai, India"):
    try:
        print(f"Fetching historical weather data for {city}...")
    except UnicodeEncodeError:
        safe_city = city.encode('ascii', errors='replace').decode('ascii')
        print(f"Fetching historical weather data for {safe_city}...")
    
    # Historical archive API has a delay of about 2-5 days, so set end_date to 5 days ago to be safe
    end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5 + 5*365)).strftime("%Y-%m-%d")
    
    print(f"Date range: {start_date} to {end_date}")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max,pressure_msl_mean",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return None
    
    if "daily" not in data:
        print("Error: 'daily' key not found in API response.")
        print(f"API Response: {data}")
        return None
        
    daily_data = data["daily"]
    df = pd.DataFrame(daily_data)
    
    # Rename columns to standardized names
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
    print(f"Successfully fetched {len(df)} rows.")
    return df

def main():
    os.makedirs("data", exist_ok=True)
    df = fetch_weather_data()
    if df is not None:
        raw_path = os.path.join("data", "raw_weather.csv")
        df.to_csv(raw_path, index=False)
        print(f"Raw data saved to {raw_path}")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()
