import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    from train_lstm import WeatherLSTM, WeatherGRU, WeatherBiLSTM, create_sequences
except ModuleNotFoundError:
    from src.train_lstm import WeatherLSTM, WeatherGRU, WeatherBiLSTM, create_sequences

def evaluate_models(data_path="data/processed_weather.csv", model_dir="models", output_dir="outputs"):
    print("Evaluating models...")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist. Run feature engineering first.")
        return False
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Evaluate Random Forest
    rf_model_path = os.path.join(model_dir, "random_forest_model.joblib")
    if not os.path.exists(rf_model_path):
        print(f"Error: {rf_model_path} not found.")
        return False
    
    rf_model = joblib.load(rf_model_path)
    feature_cols_rf = [col for col in df.columns if col not in ['date', 'target_temp']]
    
    split_idx = int(len(df) * 0.8)
    X_test_rf = df[feature_cols_rf].iloc[split_idx:]
    y_test_rf = df['target_temp'].iloc[split_idx:]
    dates_test = df['date'].iloc[split_idx:]
    
    rf_pred = rf_model.predict(X_test_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_pred))
    rf_mae = mean_absolute_error(y_test_rf, rf_pred)
    
    print("\n--- Random Forest Results ---")
    print(f"RMSE: {rf_rmse:.4f} °C")
    print(f"MAE:  {rf_mae:.4f} °C")
    
    # 2. Evaluate LSTM
    lstm_model_path = os.path.join(model_dir, "lstm_model.pth")
    lstm_scaler_path = os.path.join(model_dir, "lstm_scaler.joblib")
    
    if not (os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path)):
        print("Error: LSTM model or scaler metadata not found.")
        return False
        
    lstm_metadata = joblib.load(lstm_scaler_path)
    feature_scaler = lstm_metadata['feature_scaler']
    target_scaler = lstm_metadata['target_scaler']
    feature_cols_lstm = lstm_metadata['feature_cols']
    seq_length = lstm_metadata['seq_length']
    model_type = lstm_metadata.get('model_type', 'lstm')
    
    features_lstm = df[feature_cols_lstm].values
    targets_lstm = df['target_temp'].values
    
    # Prepare sequence test data
    test_features_lstm = features_lstm[split_idx - seq_length + 1:]
    test_targets_lstm = targets_lstm[split_idx - seq_length + 1:]
    
    test_features_scaled = feature_scaler.transform(test_features_lstm)
    test_targets_scaled = target_scaler.transform(test_targets_lstm.reshape(-1, 1)).flatten()
    
    X_test_seq, y_test_seq = create_sequences(test_features_scaled, test_targets_scaled, seq_length)
    
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_type_lower = model_type.lower()
    if model_type_lower == "gru":
        lstm_model = WeatherGRU(input_size=len(feature_cols_lstm), hidden_size=64, num_layers=2, output_size=1).to(device)
    elif model_type_lower == "bilstm":
        lstm_model = WeatherBiLSTM(input_size=len(feature_cols_lstm), hidden_size=64, num_layers=2, output_size=1).to(device)
    else:
        lstm_model = WeatherLSTM(input_size=len(feature_cols_lstm), hidden_size=64, num_layers=2, output_size=1).to(device)
        
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.eval()
    
    with torch.no_grad():
        lstm_pred_scaled = lstm_model(X_test_tensor.to(device)).cpu().numpy()
        
    lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled).flatten()
    y_actual_lstm = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    
    lstm_rmse = np.sqrt(mean_squared_error(y_actual_lstm, lstm_pred))
    lstm_mae = mean_absolute_error(y_actual_lstm, lstm_pred)
    
    print(f"\n--- {model_type.upper()} Results ---")
    print(f"RMSE: {lstm_rmse:.4f} °C")
    print(f"MAE:  {lstm_mae:.4f} °C")
    
    # Save a comparison summary table
    summary_df = pd.DataFrame({
        'Model': ['Random Forest (Baseline)', f'PyTorch {model_type.upper()}'],
        'Test RMSE (°C)': [rf_rmse, lstm_rmse],
        'Test MAE (°C)': [rf_mae, lstm_mae]
    })
    summary_df.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)
    print("\nMetrics comparison saved to outputs/metrics_comparison.csv:")
    print(summary_df.to_string(index=False))
    
    # Determine the best model and save it to a text file
    best_model = "Random Forest" if rf_rmse < lstm_rmse else f"PyTorch {model_type.upper()}"
    with open(os.path.join(model_dir, "best_model.txt"), "w", encoding='utf-8') as f:
        f.write(best_model)
    print(f"\nBest model is {best_model} based on RMSE.")
    
    # 3. Plots
    print("\nGenerating evaluation plots...")
    
    # Plot Actual vs Predicted for both models
    plt.figure(figsize=(14, 7))
    plot_days = min(120, len(dates_test))
    
    plt.plot(dates_test.iloc[-plot_days:].values, y_test_rf.iloc[-plot_days:].values, label='Actual Temp', color='black', linewidth=2.0)
    plt.plot(dates_test.iloc[-plot_days:].values, rf_pred[-plot_days:], label='Random Forest (Pred)', color='dodgerblue', linestyle='--', linewidth=1.5)
    plt.plot(dates_test.iloc[-plot_days:].values, lstm_pred[-plot_days:], label='LSTM (Pred)', color='crimson', linestyle='-.', linewidth=1.5)
    
    plt.title(f'Actual vs Predicted Next-Day Mean Temperature (Last {plot_days} Days)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved actual vs predicted plot to {plot_path}")
    
    # Plot Feature Importance for Random Forest
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = min(15, len(feature_cols_rf))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    # Avoid warning by setting hue equal to y
    y_labels = [feature_cols_rf[i] for i in top_indices]
    sns.barplot(x=importances[top_indices], y=y_labels, hue=y_labels, palette='viridis', legend=False)
    plt.title(f'Random Forest Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    fi_plot_path = os.path.join(output_dir, "rf_feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importances plot to {fi_plot_path}")
    
    return True

if __name__ == "__main__":
    evaluate_models()
