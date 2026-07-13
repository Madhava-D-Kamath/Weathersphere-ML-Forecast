import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Define LSTM Model
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

# Define GRU Model
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

# Define Bidirectional LSTM Model
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

def create_sequences(features, targets, seq_length=7):
    X, y = [], []
    for i in range(len(features) - seq_length + 1):
        X.append(features[i : i + seq_length])
        y.append(targets[i + seq_length - 1])
    return np.array(X), np.array(y)

def train_lstm(data_path="data/processed_weather.csv", model_dir="models", epochs=50, batch_size=32, seq_length=7, model_type="lstm", lr=0.001, epoch_callback=None):
    print(f"Training PyTorch {model_type.upper()} Regressor...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist. Run feature engineering first.")
        return None, None, None
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Base features to use in sequence
    feature_cols = ['temp_mean', 'temp_max', 'temp_min', 'humidity', 'precipitation', 'wind_speed', 'pressure', 'sin_month', 'cos_month', 'sin_day', 'cos_day']
    
    features = df[feature_cols].values
    targets = df['target_temp'].values
    
    # Split indexes
    split_idx = int(len(df) * 0.8)
    
    # Separate train/test before sequence generation to avoid contamination
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    
    # For test set, we need the last (seq_length - 1) items of training features to build sequences correctly
    test_features = features[split_idx - seq_length + 1:]
    test_targets = targets[split_idx - seq_length + 1:]
    dates_test = df['date'].iloc[split_idx:]
    
    # Scale features and targets
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    train_features_scaled = feature_scaler.fit_transform(train_features)
    train_targets_scaled = target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
    
    test_features_scaled = feature_scaler.transform(test_features)
    test_targets_scaled = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train, y_train = create_sequences(train_features_scaled, train_targets_scaled, seq_length)
    X_test, y_test = create_sequences(test_features_scaled, test_targets_scaled, seq_length)
    
    print(f"{model_type.upper()} Train sequences shape: {X_train.shape}")
    print(f"{model_type.upper()} Test sequences shape: {X_test.shape}")
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    model_type_lower = model_type.lower()
    if model_type_lower == "gru":
        model = WeatherGRU(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=1).to(device)
    elif model_type_lower == "bilstm":
        model = WeatherBiLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=1).to(device)
    else:
        model = WeatherLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=1).to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    model.train()
    print(f"Beginning {model_type.upper()} training loop...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")
        if epoch_callback is not None:
            epoch_callback(epoch, epochs, epoch_loss)
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        y_pred_scaled = model(X_test_device).cpu().numpy()
        
    # Inverse transform predictions and targets
    y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    test_mae = mean_absolute_error(y_actual, y_pred)
    
    print(f"\n--- {model_type.upper()} Evaluation ---")
    print(f"Test RMSE: {test_rmse:.4f} °C, Test MAE: {test_mae:.4f} °C")
    
    # Save the model and scalers
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lstm_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save scalers & features for pipeline use
    joblib.dump({
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'seq_length': seq_length,
        'model_type': model_type
    }, os.path.join(model_dir, "lstm_scaler.joblib"))
    
    print(f"{model_type.upper()} model weights saved to {model_path}")
    print(f"{model_type.upper()} scaler and metadata saved to {os.path.join(model_dir, 'lstm_scaler.joblib')}")
    
    predictions_df = pd.DataFrame({
        'date': dates_test,
        'actual': y_actual,
        'predicted': y_pred
    })
    
    return model, predictions_df, {'rmse': test_rmse, 'mae': test_mae}

if __name__ == "__main__":
    train_lstm()
