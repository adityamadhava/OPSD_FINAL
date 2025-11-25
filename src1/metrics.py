"""
MASE, sMAPE, MSE, RMSE, MAPE, coverage helpers.
Renamed from evaluation.py to match preferred repository layout.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 24) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonality: Seasonal period (24 for hourly daily seasonality)
    
    Returns:
        MASE value
    """
    # Calculate naive seasonal forecast error
    if len(y_train) < seasonality:
        # Fallback to simple naive forecast
        naive_errors = np.abs(np.diff(y_train))
    else:
        # Seasonal naive: use value from same hour in previous day
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    
    if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
        return np.nan
    
    mae = np.mean(np.abs(y_true - y_pred))
    scale = np.mean(naive_errors)
    
    return mae / scale if scale > 0 else np.nan


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        sMAPE value (as percentage)
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator > 0
    if np.sum(mask) == 0:
        return np.nan
    
    smape = np.mean(numerator[mask] / denominator[mask]) * 100
    return smape


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)."""
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Returns:
        MAPE value (as percentage)
    """
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def calculate_pi_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> float:
    """
    Calculate prediction interval coverage.
    
    Args:
        y_true: True values
        lo: Lower bound of prediction interval
        hi: Upper bound of prediction interval
        alpha: Significance level (0.2 for 80% PI)
    
    Returns:
        Coverage percentage
    """
    # Check if intervals are available
    if np.all(np.isnan(lo)) or np.all(np.isnan(hi)):
        return np.nan
    
    # Count how many true values fall within the interval
    within_interval = (y_true >= lo) & (y_true <= hi)
    coverage = np.mean(within_interval) * 100
    
    return coverage

