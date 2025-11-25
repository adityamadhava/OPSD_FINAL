"""
Evaluation metrics for time series forecasting.
Calculates MASE, sMAPE, MSE, RMSE, MAPE, and 80% PI coverage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


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


def evaluate_forecasts(
    forecasts_df: pd.DataFrame,
    train_df: pd.DataFrame,
    seasonality: int = 24
) -> dict:
    """
    Calculate all evaluation metrics for forecasts.
    
    Args:
        forecasts_df: DataFrame with columns: y_true, yhat, lo, hi
        train_df: Training data for MASE calculation
        seasonality: Seasonal period
    
    Returns:
        Dictionary with all metrics
    """
    # Extract arrays
    y_true = forecasts_df['y_true'].values
    y_pred = forecasts_df['yhat'].values
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'MASE': np.nan,
            'sMAPE': np.nan,
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'PI_coverage_80': np.nan,
            'n_forecasts': 0
        }
    
    # Calculate metrics
    y_train = train_df['load'].values
    
    metrics = {
        'MASE': calculate_mase(y_true, y_pred, y_train, seasonality),
        'sMAPE': calculate_smape(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'n_forecasts': len(y_true)
    }
    
    # PI coverage (if available)
    if 'lo' in forecasts_df.columns and 'hi' in forecasts_df.columns:
        lo = forecasts_df['lo'].values[mask]
        hi = forecasts_df['hi'].values[mask]
        metrics['PI_coverage_80'] = calculate_pi_coverage(y_true, lo, hi, alpha=0.2)
    else:
        metrics['PI_coverage_80'] = np.nan
    
    return metrics


def evaluate_country(
    country_code: str,
    output_dir: Path,
    split: str = 'dev'
) -> dict:
    """
    Evaluate forecasts for a single country and split.
    
    Args:
        country_code: Country code
        output_dir: Output directory
        split: 'dev' or 'test'
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {country_code} - {split.upper()} set")
    print(f"{'='*60}")
    
    # Load forecasts
    forecast_file = output_dir / country_code / f"{country_code}_forecasts_{split}.csv"
    if not forecast_file.exists():
        print(f"  ✗ Forecast file not found: {forecast_file}")
        return None
    
    forecasts_df = pd.read_csv(forecast_file)
    forecasts_df['timestamp'] = pd.to_datetime(forecasts_df['timestamp'])
    
    # Load training data (for MASE)
    train_file = output_dir / country_code / "train.csv"
    train_df = pd.read_csv(train_file)
    train_df.columns = train_df.columns.str.strip()
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    
    print(f"\n  Loaded {len(forecasts_df)} forecast points")
    
    # Calculate metrics
    print("\n  Calculating metrics...")
    metrics = evaluate_forecasts(forecasts_df, train_df, seasonality=24)
    
    # Print results
    print(f"\n  Results:")
    print(f"    MASE:           {metrics['MASE']:.4f}")
    print(f"    sMAPE:          {metrics['sMAPE']:.4f}%")
    print(f"    MSE:            {metrics['MSE']:.2f}")
    print(f"    RMSE:           {metrics['RMSE']:.2f}")
    print(f"    MAPE:           {metrics['MAPE']:.4f}%")
    print(f"    PI Coverage:    {metrics['PI_coverage_80']:.2f}%")
    print(f"    N Forecasts:    {metrics['n_forecasts']}")
    
    return metrics


def main():
    """Evaluate forecasts for all countries and splits."""
    output_dir = Path(__file__).parent.parent / "outputs"
    
    # =================================================================
    # SELECT COUNTRIES TO PROCESS
    # Uncomment the countries you want to run
    # =================================================================
    countries = [
        'DE',  # Germany
        # 'FR',  # France
        # 'ES',  # Spain
    ]
    splits = ['dev', 'test']
    
    all_results = []
    
    for country in countries:
        for split in splits:
            forecast_file = output_dir / country / f"{country}_forecasts_{split}.csv"
            
            # Skip if file doesn't exist
            if not forecast_file.exists():
                print(f"\nSkipping {country} - {split} (file not found)")
                continue
            
            metrics = evaluate_country(country, output_dir, split)
            if metrics:
                metrics['country'] = country
                metrics['split'] = split
                all_results.append(metrics)
    
    # Create summary DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns
        cols = ['country', 'split', 'MASE', 'sMAPE', 'MSE', 'RMSE', 'MAPE', 'PI_coverage_80', 'n_forecasts']
        results_df = results_df[cols]
        
        # Save results
        results_file = output_dir / "evaluation_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n{'='*60}")
        print("Evaluation Summary")
        print(f"{'='*60}")
        print(f"\n{results_df.to_string(index=False)}")
        print(f"\nSaved results to: {results_file}")
        
        # Create comparison table for test set
        test_results = results_df[results_df['split'] == 'test']
        if len(test_results) > 0:
            print(f"\n{'='*60}")
            print("Test Set Comparison (All Countries)")
            print(f"{'='*60}")
            print(f"\n{test_results[['country', 'MASE', 'sMAPE', 'RMSE', 'MAPE', 'PI_coverage_80']].to_string(index=False)}")
    else:
        print("\nNo forecast files found to evaluate.")


if __name__ == "__main__":
    main()

