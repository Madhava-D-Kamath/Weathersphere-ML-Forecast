import os
import sys

# Add current directory to the python path to make imports work seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import fetch_weather_data
from feature_engineering import engineer_features
from train_random_forest import train_rf
from train_lstm import train_lstm
from evaluate import evaluate_models

def run_pipeline():
    print("=" * 60)
    print("WEATHERSPHERE AI PIPELINE: STARTING END-TO-END EXECUTION")
    print("=" * 60)
    
    # Step 1: Ingestion
    print("\n--- STEP 1: DATA INGESTION ---")
    raw_path = "data/raw_weather.csv"
    os.makedirs("data", exist_ok=True)
    df_raw = fetch_weather_data()
    if df_raw is None:
        print("Data Ingestion failed. Exiting pipeline.")
        sys.exit(1)
    df_raw.to_csv(raw_path, index=False)
    print(f"Data Ingestion completed. Saved to {raw_path}")
    
    # Step 2: Feature Engineering
    print("\n--- STEP 2: FEATURE ENGINEERING ---")
    processed_path = "data/processed_weather.csv"
    success = engineer_features(raw_path, processed_path)
    if not success:
        print("Feature Engineering failed. Exiting pipeline.")
        sys.exit(1)
    print("Feature Engineering completed.")
    
    # Step 3: Train Random Forest
    print("\n--- STEP 3: TRAINING RANDOM FOREST REGRESSOR ---")
    rf_model, rf_preds, rf_metrics = train_rf(processed_path)
    if rf_model is None:
        print("Random Forest Training failed. Exiting pipeline.")
        sys.exit(1)
    print("Random Forest training completed.")
    
    # Step 4: Train LSTM
    print("\n--- STEP 4: TRAINING PYTORCH LSTM REGRESSOR ---")
    # Training for 50 epochs balances speed and convergence
    lstm_model, lstm_preds, lstm_metrics = train_lstm(processed_path, epochs=50)
    if lstm_model is None:
        print("LSTM Training failed. Exiting pipeline.")
        sys.exit(1)
    print("LSTM training completed.")
    
    # Step 5: Evaluation & Visualization
    print("\n--- STEP 5: EVALUATION AND VISUALIZATION ---")
    success = evaluate_models(processed_path)
    if not success:
        print("Evaluation failed. Exiting pipeline.")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("WEATHERSPHERE AI PIPELINE: PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
