import os
import pandas as pd
import numpy as np

def engineer_features(input_path="data/raw_weather.csv", output_path="data/processed_weather.csv"):
    print("Loading raw weather data for feature engineering...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Run data ingestion first.")
        return False
        
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Target variable: next-day mean temperature
    df['target_temp'] = df['temp_mean'].shift(-1)
    
    # Feature columns to create lag and rolling statistics for
    feature_cols = ['temp_mean', 'temp_max', 'temp_min', 'humidity', 'precipitation', 'wind_speed', 'pressure']
    
    # 1. Create Lag Features (1, 3, 7 days)
    print("Creating lag features (1, 3, 7 days)...")
    for lag in [1, 3, 7]:
        for col in feature_cols:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    # 2. Create Rolling Averages (7-day, 30-day) for temp_mean and other core columns
    print("Creating rolling average features (7-day, 30-day)...")
    for window in [7, 30]:
        df[f'temp_mean_roll_mean_{window}'] = df['temp_mean'].rolling(window=window).mean()
        df[f'humidity_roll_mean_{window}'] = df['humidity'].rolling(window=window).mean()
        df[f'pressure_roll_mean_{window}'] = df['pressure'].rolling(window=window).mean()
        
    # 3. Seasonal Features
    print("Creating seasonal encoding features...")
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Sin/Cos encoding for month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Sin/Cos encoding for day of year
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # 4. Handle Missing Values
    # Shifting and rolling introduces missing values
    # In time series, dropping rows with missing features/targets is standard
    print("Handling missing values...")
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {initial_len - len(df)} rows due to lag/rolling/shift operations. {len(df)} rows remaining.")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return True

if __name__ == "__main__":
    engineer_features()
