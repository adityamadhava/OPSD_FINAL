"""
Visualize anomaly detection results.
Shows time series, residuals, z-scores, CUSUM, and ML predictions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_anomalies(
    df: pd.DataFrame,
    country_code: str,
    output_dir: Path,
    split: str = 'dev',
    max_points: int = 2000
):
    """
    Create comprehensive anomaly visualization plots.
    
    Args:
        df: DataFrame with anomaly detection results
        country_code: Country code
        output_dir: Output directory
        split: 'dev' or 'test'
        max_points: Maximum points to plot (for performance)
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Sample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step].copy()
    else:
        df_plot = df.copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 14))
    fig.suptitle(f'{country_code} - Anomaly Detection Visualization ({split.upper()})', fontsize=16)
    
    # 1. Time series with anomalies
    ax = axes[0]
    ax.plot(df_plot['timestamp'], df_plot['y_true'], 'b-', label='Actual', linewidth=1, alpha=0.7)
    ax.plot(df_plot['timestamp'], df_plot['yhat'], 'g--', label='Forecast', linewidth=1, alpha=0.7)
    
    # Highlight anomalies
    anomalies_z = df_plot[df_plot['flag_z'] == 1]
    if len(anomalies_z) > 0:
        ax.scatter(anomalies_z['timestamp'], anomalies_z['y_true'], 
                  c='red', s=20, alpha=0.6, label='Z-score Anomaly', marker='o')
    
    if 'flag_cusum' in df_plot.columns:
        anomalies_c = df_plot[df_plot['flag_cusum'] == 1]
        if len(anomalies_c) > 0:
            ax.scatter(anomalies_c['timestamp'], anomalies_c['y_true'],
                      c='orange', s=20, alpha=0.6, label='CUSUM Anomaly', marker='^')
    
    ax.set_ylabel('Load (MW)')
    ax.set_title('Time Series with Anomalies')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Residuals
    ax = axes[1]
    ax.plot(df_plot['timestamp'], df_plot['residual'], 'b-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    # Highlight anomalies
    if len(anomalies_z) > 0:
        ax.scatter(anomalies_z['timestamp'], anomalies_z['residual'],
                  c='red', s=20, alpha=0.6, marker='o')
    
    ax.set_ylabel('Residual')
    ax.set_title('Residuals (y_true - yhat)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Z-scores
    ax = axes[2]
    ax.plot(df_plot['timestamp'], df_plot['z_resid'], 'b-', linewidth=1, alpha=0.7)
    ax.axhline(y=3.0, color='r', linestyle='--', linewidth=1, label='Threshold ±3.0')
    ax.axhline(y=-3.0, color='r', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Highlight anomalies
    if len(anomalies_z) > 0:
        ax.scatter(anomalies_z['timestamp'], anomalies_z['z_resid'],
                  c='red', s=20, alpha=0.6, marker='o')
    
    ax.set_ylabel('Z-Score')
    ax.set_title('Rolling Z-Score of Residuals')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. CUSUM
    if 'cusum_max' in df_plot.columns:
        ax = axes[3]
        ax.plot(df_plot['timestamp'], df_plot['cusum_max'], 'b-', linewidth=1, alpha=0.7)
        ax.axhline(y=5.0, color='r', linestyle='--', linewidth=1, label='Threshold 5.0')
        
        anomalies_c = df_plot[df_plot['flag_cusum'] == 1]
        if len(anomalies_c) > 0:
            ax.scatter(anomalies_c['timestamp'], anomalies_c['cusum_max'],
                      c='orange', s=20, alpha=0.6, marker='^')
        
        ax.set_ylabel('CUSUM Max')
        ax.set_title('CUSUM Values')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[3].axis('off')
    
    # 5. ML Predictions (if available)
    if 'is_anomaly_ml' in df_plot.columns or 'anomaly_probability' in df_plot.columns:
        ax = axes[4]
        
        if 'anomaly_probability' in df_plot.columns:
            ax.plot(df_plot['timestamp'], df_plot['anomaly_probability'], 
                   'purple', linewidth=1, alpha=0.7, label='Anomaly Probability')
            ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Threshold 0.5')
            ax.set_ylabel('Probability')
            ax.set_ylim([0, 1])
        else:
            anomalies_ml = df_plot[df_plot['is_anomaly_ml'] == 1]
            if len(anomalies_ml) > 0:
                ax.scatter(anomalies_ml['timestamp'], 
                          [1] * len(anomalies_ml),
                          c='purple', s=20, alpha=0.6, marker='s', label='ML Anomaly')
            ax.set_ylabel('Anomaly Flag')
            ax.set_ylim([-0.1, 1.1])
        
        ax.set_xlabel('Timestamp')
        ax.set_title('ML Classifier Predictions')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[4].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    output_file = country_dir / f"anomalies_visualization_{split}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_file}")


def main():
    """Visualize anomalies for selected country."""
    output_dir = Path(__file__).parent.parent / "outputs"
    
    # =================================================================
    # SELECT COUNTRY AND SPLIT TO PROCESS
    # Uncomment the country you want to run
    # =================================================================
    country = 'DE'  # Germany
    country = 'FR'  # France
    country = 'ES'  # Spain
    
    split = 'dev'  # or 'test'
    
    # Load anomalies file
    anomalies_file = output_dir / country / f"{country}_anomalies.csv"
    
    if not anomalies_file.exists():
        print(f"Anomaly file not found: {anomalies_file}")
        print("Please run anomaly detection first: python src/anomaly_detection.py")
        return
    
    print(f"Loading anomalies from {anomalies_file}")
    df = pd.read_csv(anomalies_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Also load forecasts for yhat (required for visualization)
    forecast_file = output_dir / country / f"{country}_forecasts_{split}.csv"
    if forecast_file.exists():
        print(f"Loading forecasts from {forecast_file}")
        forecasts = pd.read_csv(forecast_file)
        forecasts.columns = forecasts.columns.str.strip()
        forecasts['timestamp'] = pd.to_datetime(forecasts['timestamp'])
        
        # Merge forecasts
        df = df.merge(forecasts[['timestamp', 'yhat']], on='timestamp', how='left', suffixes=('', '_forecast'))
        
        # If merge didn't work, try direct assignment
        if 'yhat' not in df.columns and 'yhat_forecast' in df.columns:
            df['yhat'] = df['yhat_forecast']
            df = df.drop(columns=['yhat_forecast'])
    else:
        print(f"Warning: Forecast file not found: {forecast_file}")
        print("Visualization will be limited without yhat values")
        df['yhat'] = np.nan
    
    # Check if yhat exists, if not create dummy
    if 'yhat' not in df.columns:
        print("Warning: yhat column not found, using y_true as placeholder")
        df['yhat'] = df['y_true']
    
    # Calculate residual if not present
    if 'residual' not in df.columns:
        df['residual'] = df['y_true'] - df['yhat']
    
    # Create visualization
    plot_anomalies(df, country, output_dir, split=split)
    
    print(f"\n{'='*60}")
    print("Anomaly Visualization Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

