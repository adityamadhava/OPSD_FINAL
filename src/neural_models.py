"""
GRU/LSTM Neural Network Models for Multi-Horizon Forecasting
Direct multi-horizon: last 168h → next 24 steps
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
from typing import Tuple, Optional
import gc

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

sys.path.insert(0, str(Path(__file__).parent.parent))


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Array of shape (n_samples, seq_len, n_features)
            targets: Array of shape (n_samples, horizon)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class GRUForecaster(nn.Module):
    """GRU-based multi-horizon forecaster."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 24
    ):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, horizon)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        # Take last output
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_size)
        forecast = self.fc(last_hidden)  # (batch, horizon)
        return forecast


class LSTMForecaster(nn.Module):
    """LSTM-based multi-horizon forecaster."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 24
    ):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, horizon)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        forecast = self.fc(last_hidden)  # (batch, horizon)
        return forecast


def create_sequences(
    data: np.ndarray,
    seq_len: int = 168,
    horizon: int = 24,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for training.
    
    Args:
        data: 1D array of time series values
        seq_len: Length of input sequence (168h = 7 days)
        horizon: Length of forecast horizon (24h = 1 day)
        stride: Step size for creating sequences
    
    Returns:
        X: Sequences of shape (n_samples, seq_len, 1)
        y: Targets of shape (n_samples, horizon)
    """
    X, y = [], []
    
    for i in range(0, len(data) - seq_len - horizon + 1, stride):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to (n_samples, seq_len, 1) for univariate
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize data using min-max scaling."""
    data_min = data.min()
    data_max = data.max()
    
    if data_max == data_min:
        return data, data_min, data_max
    
    normalized = (data - data_min) / (data_max - data_min)
    return normalized, data_min, data_max


def denormalize_data(
    data: np.ndarray,
    data_min: float,
    data_max: float
) -> np.ndarray:
    """Denormalize data."""
    return data * (data_max - data_min) + data_min


def train_neural_model(
    train_data: pd.Series,
    model_type: str = 'gru',
    seq_len: int = 168,
    horizon: int = 24,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[str] = None
) -> Tuple[nn.Module, dict]:
    """
    Train a GRU or LSTM model.
    
    Args:
        train_data: Training time series
        model_type: 'gru' or 'lstm'
        seq_len: Input sequence length (168h)
        horizon: Forecast horizon (24h)
        hidden_size: Hidden layer size
        num_layers: Number of RNN layers
        dropout: Dropout rate
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        Trained model and normalization parameters
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural models. Install with: pip install torch")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"  Training {model_type.upper()} model on {device}")
    print(f"  Sequence length: {seq_len}h, Horizon: {horizon}h")
    
    # Prepare data
    data_array = train_data.values.astype(np.float32)
    
    # Normalize
    data_norm, data_min, data_max = normalize_data(data_array)
    
    # Create sequences
    X, y = create_sequences(data_norm, seq_len=seq_len, horizon=horizon, stride=1)
    
    print(f"  Created {len(X)} training sequences")
    
    # Create dataset and dataloader
    dataset = TimeSeriesDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    if model_type.lower() == 'gru':
        model = GRUForecaster(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )
    elif model_type.lower() == 'lstm':
        model = LSTMForecaster(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'gru' or 'lstm'")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    print(f"  Training complete. Final loss: {train_losses[-1]:.6f}")
    
    # Store normalization params
    norm_params = {
        'data_min': data_min,
        'data_max': data_max
    }
    
    return model, norm_params


def forecast_neural_model(
    model: nn.Module,
    last_sequence: np.ndarray,
    norm_params: dict,
    horizon: int = 24,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate forecast using trained neural model.
    
    Args:
        model: Trained model
        last_sequence: Last seq_len values from history
        norm_params: Normalization parameters
        horizon: Forecast horizon
        device: 'cpu' or 'cuda'
    
    Returns:
        yhat: Forecast mean
        lo: Lower bound (80% PI) - approximated
        hi: Upper bound (80% PI) - approximated
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    
    # Normalize input
    data_min = norm_params['data_min']
    data_max = norm_params['data_max']
    last_sequence_norm = (last_sequence - data_min) / (data_max - data_min)
    
    # Reshape to (1, seq_len, 1)
    if last_sequence_norm.ndim == 1:
        last_sequence_norm = last_sequence_norm.reshape(1, -1, 1)
    
    # Convert to tensor
    X = torch.FloatTensor(last_sequence_norm).to(device)
    
    # Forecast
    with torch.no_grad():
        forecast_norm = model(X).cpu().numpy().flatten()
    
    # Denormalize
    yhat = denormalize_data(forecast_norm, data_min, data_max)
    
    # Approximate prediction intervals (simple approach: ±10% with some variance)
    # In practice, you might use quantile regression or ensemble methods
    std_estimate = np.std(yhat) * 0.1  # Rough estimate
    z_score = 1.28  # For 80% PI (two-tailed)
    
    lo = yhat - z_score * std_estimate
    hi = yhat + z_score * std_estimate
    
    return yhat, lo, hi


def backtest_neural_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str = 'gru',
    seq_len: int = 168,
    horizon: int = 24,
    stride: int = 24,
    warmup_days: int = 60,
    retrain_every: int = 24,
    device: Optional[str] = None
) -> pd.DataFrame:
    """
    Backtest neural model with expanding origin.
    
    Args:
        train_df: Training data
        test_df: Test data
        model_type: 'gru' or 'lstm'
        seq_len: Input sequence length (168h)
        horizon: Forecast horizon (24h)
        stride: Step size between forecasts
        warmup_days: Minimum days of history needed
        retrain_every: Retrain model every N hours (0 = never retrain)
        device: 'cpu' or 'cuda'
    
    Returns:
        DataFrame with forecasts
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural models. Install with: pip install torch")
    
    print(f"\n{'='*60}")
    print(f"Neural Model Backtesting ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"  Sequence length: {seq_len}h, Horizon: {horizon}h, Stride: {stride}h")
    
    # Prepare data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    train_series = train_df.set_index('timestamp')['load']
    test_series = test_df.set_index('timestamp')['load']
    
    # Combine for expanding window
    all_data = pd.concat([train_series, test_series])
    
    # Initialize
    warmup_hours = warmup_days * 24
    forecasts = []
    model = None
    norm_params = None
    last_retrain_idx = -1
    
    # Get forecast points
    test_start_idx = len(train_series)
    test_end_idx = len(all_data)
    
    forecast_points = list(range(test_start_idx, test_end_idx - horizon + 1, stride))
    
    print(f"  Total forecast points: {len(forecast_points)}")
    print(f"  Warm-up hours: {warmup_hours}")
    
    for i, forecast_idx in enumerate(forecast_points):
        if (i + 1) % 10 == 0:
            print(f"    Processing forecast {i+1}/{len(forecast_points)}...", end='\r')
        
        # Get history up to forecast point
        history = all_data.iloc[:forecast_idx]
        
        # Check if we need to retrain
        should_retrain = (
            model is None or
            (retrain_every > 0 and (forecast_idx - last_retrain_idx) >= retrain_every) or
            len(history) < warmup_hours
        )
        
        if should_retrain and len(history) >= warmup_hours:
            # Use last seq_len + some buffer for training
            train_window = min(len(history), 365 * 24)  # Max 1 year
            train_data = history.iloc[-train_window:]
            
            try:
                model, norm_params = train_neural_model(
                    train_data,
                    model_type=model_type,
                    seq_len=seq_len,
                    horizon=horizon,
                    epochs=30,  # Fewer epochs for faster retraining
                    device=device
                )
                last_retrain_idx = forecast_idx
            except Exception as e:
                print(f"\n    Warning: Training failed at {forecast_idx}: {e}")
                if model is None:
                    continue  # Skip if no model available
        
        if model is None:
            continue
        
        # Get last sequence
        if len(history) < seq_len:
            continue
        
        last_seq = history.iloc[-seq_len:].values
        
        # Forecast
        try:
            yhat, lo, hi = forecast_neural_model(
                model,
                last_seq,
                norm_params,
                horizon=horizon,
                device=device
            )
        except Exception as e:
            print(f"\n    Warning: Forecast failed at {forecast_idx}: {e}")
            yhat = np.full(horizon, np.nan)
            lo = np.full(horizon, np.nan)
            hi = np.full(horizon, np.nan)
        
        # Get actual values
        actual_start = forecast_idx
        actual_end = min(forecast_idx + horizon, len(all_data))
        actual_values = all_data.iloc[actual_start:actual_end].values
        
        # Pad if needed
        if len(actual_values) < horizon:
            actual_values = np.pad(
                actual_values,
                (0, horizon - len(actual_values)),
                mode='constant',
                constant_values=np.nan
            )
        
        # Create forecast timestamps
        forecast_timestamp = all_data.index[forecast_idx]
        forecast_timestamps = pd.date_range(
            start=forecast_timestamp,
            periods=horizon,
            freq='H'
        )
        
        # Store results
        for h in range(horizon):
            forecasts.append({
                'timestamp': forecast_timestamps[h],
                'y_true': actual_values[h] if h < len(actual_values) else np.nan,
                'yhat': yhat[h] if h < len(yhat) else np.nan,
                'lo': lo[h] if h < len(lo) else np.nan,
                'hi': hi[h] if h < len(hi) else np.nan,
                'horizon': h + 1,
                'train_end': forecast_timestamp
            })
        
        # Cleanup GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n  Completed {len(forecast_points)} forecasts")
    
    forecasts_df = pd.DataFrame(forecasts)
    return forecasts_df

