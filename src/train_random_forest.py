import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

def train_rf(data_path="data/processed_weather.csv", model_dir="models"):
    print("Training Random Forest Regressor...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist. Run feature engineering first.")
        return None, None, None
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Exclude columns that are not features (date, target_temp)
    feature_cols = [col for col in df.columns if col not in ['date', 'target_temp']]
    
    X = df[feature_cols]
    y = df['target_temp']
    
    # Time-based split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df['date'].iloc[split_idx:]
    
    print(f"Train set size: {len(X_train)} rows ({df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[split_idx-1].strftime('%Y-%m-%d')})")
    print(f"Test set size: {len(X_test)} rows ({df['date'].iloc[split_idx].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    
    # Initialize and train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print("\n--- Random Forest Evaluation ---")
    print(f"Train RMSE: {train_rmse:.4f} °C, Train MAE: {train_mae:.4f} °C")
    print(f"Test RMSE: {test_rmse:.4f} °C, Test MAE: {test_mae:.4f} °C")
    
    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_model.joblib")
    joblib.dump(rf_model, model_path)
    print(f"Random Forest model saved to {model_path}")
    
    # Store predictions for plotting
    predictions_df = pd.DataFrame({
        'date': dates_test,
        'actual': y_test,
        'predicted': y_pred_test
    })
    
    # Return model, predictions, and metrics
    return rf_model, predictions_df, {'rmse': test_rmse, 'mae': test_mae, 'feature_names': feature_cols}

if __name__ == "__main__":
    train_rf()
