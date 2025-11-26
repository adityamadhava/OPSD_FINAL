"""
Live ingestion and online adaptation simulation.
Simulates hour-by-hour data arrival with model adaptation on drift detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.evaluation import calculate_mase, calculate_pi_coverage as calc_pi_coverage


def load_model_params(country_code: str, output_dir: Path) -> dict:
    """Load best SARIMA model parameters."""
    grid_file = output_dir / country_code / "sarima_grid_search.csv"
    if not grid_file.exists():
        raise FileNotFoundError(f"Grid search results not found: {grid_file}")
    
    df = pd.read_csv(grid_file)
    best = df.iloc[0]
    
    return {
        'p': int(best['p']),
        'd': int(best['d']),
        'q': int(best['q']),
        'P': int(best['P']),
        'D': int(best['D']),
        'Q': int(best['Q']),
        's': int(best['s'])
    }


def calculate_ewma(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Calculate Exponential Weighted Moving Average."""
    ewma = np.zeros_like(values)
    ewma[0] = values[0]
    
    for i in range(1, len(values)):
        ewma[i] = alpha * values[i] + (1 - alpha) * ewma[i-1]
    
    return ewma


def detect_drift(
    z_scores: np.ndarray,
    window_days: int = 30,
    alpha: float = 0.1,
    percentile: float = 95.0
) -> bool:
    """
    Detect drift using EWMA of |z|.
    
    Trigger if: EWMA(|z|; α=0.1) > 95th percentile of |z| over last 30 days
    
    Args:
        z_scores: Array of z-scores
        window_days: Window for percentile calculation (default 30 days = 720 hours)
        alpha: EWMA smoothing parameter (default 0.1)
        percentile: Percentile threshold (default 95.0)
    
    Returns:
        True if drift detected, False otherwise
    """
    if len(z_scores) < window_days * 24:
        return False
    
    abs_z = np.abs(z_scores)
    
    # Calculate EWMA of |z|
    ewma_abs_z = calculate_ewma(abs_z, alpha=alpha)
    
    # Get last 30 days for percentile
    last_window = abs_z[-window_days * 24:]
    threshold = np.percentile(last_window, percentile)
    
    # Check if current EWMA exceeds threshold
    current_ewma = ewma_abs_z[-1]
    
    return current_ewma > threshold


def refit_sarima_model(
    train_data: pd.Series,
    model_params: dict,
    max_iter: int = 50
) -> object:
    """
    Refit SARIMA model on training data.
    
    Args:
        train_data: Training time series
        model_params: Model parameters
        max_iter: Maximum iterations
    
    Returns:
        Fitted model
    """
    model = SARIMAX(
        train_data,
        order=(model_params['p'], model_params['d'], model_params['q']),
        seasonal_order=(model_params['P'], model_params['D'], model_params['Q'], model_params['s']),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted = model.fit(disp=False, maxiter=max_iter)
    return fitted


def calculate_rolling_metrics(
    forecasts_df: pd.DataFrame,
    train_df: pd.DataFrame,
    window_hours: int = 168  # 7 days
) -> dict:
    """
    Calculate rolling 7-day MASE and 80% PI coverage.
    
    Args:
        forecasts_df: DataFrame with forecasts
        train_df: Training data for MASE
        window_hours: Rolling window (default 168 = 7 days)
    
    Returns:
        Dictionary with metrics
    """
    if len(forecasts_df) < window_hours:
        return {'mase': np.nan, 'pi_coverage': np.nan}
    
    # Get last window_hours
    recent = forecasts_df.tail(window_hours).copy()
    
    # Calculate MASE
    y_train = train_df['load'].values
    mase = calculate_mase(
        recent['y_true'].values,
        recent['yhat'].values,
        y_train,
        seasonality=24
    )
    
    # Calculate PI coverage
    if 'lo' in recent.columns and 'hi' in recent.columns:
        pi_coverage = calc_pi_coverage(
            recent['y_true'].values,
            recent['lo'].values,
            recent['hi'].values,
            alpha=0.2
        )
    else:
        pi_coverage = np.nan
    
    return {
        'mase': mase,
        'pi_coverage': pi_coverage
    }


def simulate_live_ingestion(
    country_code: str,
    output_dir: Path,
    start_hours: int = 2000,
    adaptation_strategy: str = 'rolling_refit',
    refit_window_days: int = 90
) -> pd.DataFrame:
    """
    Simulate live ingestion with online adaptation.
    
    Args:
        country_code: Country code
        output_dir: Output directory
        start_hours: Number of hours to simulate (default 2000)
        adaptation_strategy: 'rolling_refit' or 'neural_finetune'
        refit_window_days: Days of data for refit (default 90)
    
    Returns:
        DataFrame with update logs
    """
    print(f"\n{'='*60}")
    print(f"Live Ingestion Simulation - {country_code}")
    print(f"{'='*60}")
    print(f"Strategy: {adaptation_strategy}")
    print(f"Simulating {start_hours} hours...")
    
    # Load data
    train_file = output_dir / country_code / "train.csv"
    test_file = output_dir / country_code / "test.csv"
    
    train_df = pd.read_csv(train_file)
    train_df.columns = train_df.columns.str.strip()
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    
    test_df = pd.read_csv(test_file)
    test_df.columns = test_df.columns.str.strip()
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Combine train + test for simulation
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    all_data = all_data.sort_values('timestamp').reset_index(drop=True)
    
    # Start from end of training set
    start_idx = len(train_df)
    end_idx = min(start_idx + start_hours, len(all_data))
    
    print(f"\nSimulation range: {all_data.iloc[start_idx]['timestamp']} to {all_data.iloc[end_idx-1]['timestamp']}")
    
    # Load model parameters
    model_params = load_model_params(country_code, output_dir)
    print(f"\nInitial model: SARIMA({model_params['p']},{model_params['d']},{model_params['q']})"
          f"({model_params['P']},{model_params['D']},{model_params['Q']}){model_params['s']}")
    
    # Initialize
    historical_data = train_df.copy()
    forecasts_log = []
    update_log = []
    z_scores_history = []
    current_model = None  # Will be fitted on first adaptation
    
    # Track metrics before/after updates
    metrics_before = None
    metrics_after = None
    
    print(f"\nStarting simulation...")
    print(f"Progress: ", end='', flush=True)
    
    for hour_idx in range(start_idx, end_idx):
        if (hour_idx - start_idx) % 200 == 0:
            print(f"{hour_idx - start_idx}/{start_hours} ", end='', flush=True)
        
        current_time = all_data.iloc[hour_idx]['timestamp']
        current_load = all_data.iloc[hour_idx]['load']
        
        # Append new data point
        new_row = pd.DataFrame({
            'timestamp': [current_time],
            'load': [current_load]
        })
        historical_data = pd.concat([historical_data, new_row], ignore_index=True)
        
        # Check if it's 00:00 UTC (forecast time)
        is_forecast_time = (current_time.hour == 0) and (current_time.minute == 0)
        
        # Calculate z-score for drift detection (using rolling window on residuals)
        if len(forecasts_log) >= 336:  # Need at least 14 days for rolling stats
            recent_forecasts = pd.DataFrame(forecasts_log).tail(336)
            residuals = recent_forecasts['y_true'].values - recent_forecasts['yhat'].values
            
            # Rolling z-score (window=336h, min_periods=168h)
            if len(residuals) >= 168:
                window_mean = np.mean(residuals[-336:])
                window_std = np.std(residuals[-336:]) + 1e-8
                current_residual = current_load - (recent_forecasts.iloc[-1]['yhat'] if len(recent_forecasts) > 0 else current_load)
                z_score = (current_residual - window_mean) / window_std
                z_scores_history.append(z_score)
        
        # Check for drift (every hour, after we have enough history)
        drift_detected = False
        if len(z_scores_history) >= 30 * 24:  # Need at least 30 days
            drift_detected = detect_drift(np.array(z_scores_history), window_days=30, alpha=0.1, percentile=95.0)
        
        # Adaptation triggers
        should_adapt = False
        adaptation_reason = None
        
        if is_forecast_time:
            # Scheduled refit (daily at 00:00)
            should_adapt = True
            adaptation_reason = 'scheduled'
        elif drift_detected:
            # Drift-triggered refit
            should_adapt = True
            adaptation_reason = 'drift'
        elif current_model is None:
            # Initial model fit
            should_adapt = True
            adaptation_reason = 'initial'
        
        # Perform adaptation if needed
        if should_adapt and adaptation_strategy == 'rolling_refit':
            start_time = time.time()
            
            # Get metrics before update
            if len(forecasts_log) >= 168:  # Need at least 7 days
                metrics_before = calculate_rolling_metrics(
                    pd.DataFrame(forecasts_log),
                    train_df,
                    window_hours=168
                )
            
            # Refit on last N days
            refit_window_hours = refit_window_days * 24
            if len(historical_data) > refit_window_hours:
                refit_data = historical_data.tail(refit_window_hours)['load']
            else:
                refit_data = historical_data['load']
            
            try:
                fitted_model = refit_sarima_model(refit_data, model_params)
                current_model = fitted_model
                duration_s = time.time() - start_time
                
                # Generate forecast for next 24 hours (only at 00:00 UTC)
                if is_forecast_time or adaptation_reason == 'initial':
                    forecast = fitted_model.get_forecast(steps=24)
                    yhat = forecast.predicted_mean.values
                    conf_int = forecast.conf_int()
                    lo = conf_int.iloc[:, 0].values
                    hi = conf_int.iloc[:, 1].values
                    
                    # Store forecasts
                    for h in range(24):
                        if hour_idx + h < len(all_data):
                            true_value = all_data.iloc[hour_idx + h]['load']
                            forecasts_log.append({
                                'timestamp': all_data.iloc[hour_idx + h]['timestamp'],
                                'y_true': true_value,
                                'yhat': yhat[h],
                                'lo': lo[h],
                                'hi': hi[h]
                            })
                
                # Get metrics after update
                if len(forecasts_log) >= 168:
                    metrics_after = calculate_rolling_metrics(
                        pd.DataFrame(forecasts_log),
                        train_df,
                        window_hours=168
                    )
                
                # Log update
                update_log.append({
                    'timestamp': current_time,
                    'strategy': adaptation_strategy,
                    'reason': adaptation_reason,
                    'duration_s': duration_s,
                    'mase_before': metrics_before['mase'] if metrics_before else np.nan,
                    'mase_after': metrics_after['mase'] if metrics_after else np.nan,
                    'pi_coverage_before': metrics_before['pi_coverage'] if metrics_before else np.nan,
                    'pi_coverage_after': metrics_after['pi_coverage'] if metrics_after else np.nan
                })
                
            except Exception as e:
                print(f"\nWarning: Model refit failed at {current_time}: {e}")
                duration_s = time.time() - start_time
                update_log.append({
                    'timestamp': current_time,
                    'strategy': adaptation_strategy,
                    'reason': adaptation_reason,
                    'duration_s': duration_s,
                    'mase_before': np.nan,
                    'mase_after': np.nan,
                    'pi_coverage_before': np.nan,
                    'pi_coverage_after': np.nan
                })
        
        # If it's forecast time but no adaptation, still forecast (use existing model)
        elif is_forecast_time and current_model is not None:
            try:
                forecast = current_model.get_forecast(steps=24)
                yhat = forecast.predicted_mean.values
                conf_int = forecast.conf_int()
                lo = conf_int.iloc[:, 0].values
                hi = conf_int.iloc[:, 1].values
                
                # Store forecasts
                for h in range(24):
                    if hour_idx + h < len(all_data):
                        true_value = all_data.iloc[hour_idx + h]['load']
                        forecasts_log.append({
                            'timestamp': all_data.iloc[hour_idx + h]['timestamp'],
                            'y_true': true_value,
                            'yhat': yhat[h],
                            'lo': lo[h],
                            'hi': hi[h]
                        })
            except Exception as e:
                # If forecast fails, skip
                pass
    
    print(f"\n\nSimulation complete!")
    print(f"Total updates: {len(update_log)}")
    print(f"Total forecasts: {len(forecasts_log)}")
    
    # Save update log
    if update_log:
        log_df = pd.DataFrame(update_log)
        log_file = output_dir / country_code / f"{country_code}_online_updates.csv"
        log_df.to_csv(log_file, index=False)
        print(f"\nSaved update log to {log_file}")
    
    # Save forecasts
    if forecasts_log:
        forecasts_df = pd.DataFrame(forecasts_log)
        forecasts_file = output_dir / country_code / f"{country_code}_online_forecasts.csv"
        forecasts_df.to_csv(forecasts_file, index=False)
        print(f"Saved forecasts to {forecasts_file}")
    
    return pd.DataFrame(update_log) if update_log else pd.DataFrame()


def main():
    """Run live ingestion simulation for selected country."""
    output_dir = Path(__file__).parent.parent / "outputs"
    
    # =================================================================
    # SELECT COUNTRY TO PROCESS
    # Uncomment the country you want to run
    # =================================================================
    country = 'DE'  # Germany
    # country = 'FR'  # France
    # country = 'ES'  # Spain
    
    print(f"\n{'#'*60}")
    print(f"LIVE INGESTION + ONLINE ADAPTATION SIMULATION")
    print(f"{'#'*60}")
    print(f"Country: {country}")
    print(f"Strategy: Rolling SARIMA Refit")
    print(f"Refit window: 90 days")
    print(f"Simulation hours: 100 (TEST MODE)")
    
    # Run simulation (testing with 100 hours)
    # update_log = simulate_live_ingestion(
    #     country,
    #     output_dir,
    #     start_hours=100,  # Test with 100 hours first
    #     adaptation_strategy='rolling_refit',
    #     refit_window_days=90
    # )
    update_log = simulate_live_ingestion(
        country,
        output_dir,
        start_hours=2000,  # Full simulation
        adaptation_strategy='rolling_refit',
        refit_window_days=90
    )
    
    if len(update_log) > 0:
        print(f"\n{'='*60}")
        print("Update Summary")
        print(f"{'='*60}")
        print(f"\nTotal updates: {len(update_log)}")
        print(f"Scheduled updates: {(update_log['reason'] == 'scheduled').sum()}")
        print(f"Drift-triggered updates: {(update_log['reason'] == 'drift').sum()}")
        print(f"Average duration: {update_log['duration_s'].mean():.2f}s")
        
        # Metrics comparison
        if 'mase_before' in update_log.columns:
            valid_before = update_log['mase_before'].notna()
            valid_after = update_log['mase_after'].notna()
            valid_both = valid_before & valid_after
            
            if valid_both.sum() > 0:
                print(f"\nMetrics Comparison (before vs after):")
                print(f"  MASE improvement: {update_log.loc[valid_both, 'mase_before'].mean():.4f} → "
                      f"{update_log.loc[valid_both, 'mase_after'].mean():.4f}")
                print(f"  PI Coverage: {update_log.loc[valid_both, 'pi_coverage_before'].mean():.2f}% → "
                      f"{update_log.loc[valid_both, 'pi_coverage_after'].mean():.2f}%")
    
    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


