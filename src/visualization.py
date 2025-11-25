"""
Visualization utilities for time series analysis.
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict


def plot_last_14_days(
    df: pd.DataFrame,
    country_code: str,
    output_dir: Path,
    figsize: tuple = (15, 6)
):
    """
    Plot the last 14 days of load data to confirm hourly cadence.
    
    Args:
        df: DataFrame with timestamp and load columns
        country_code: Country code for title
        output_dir: Base output directory (will create country subfolder)
        figsize: Figure size
    """
    # Get last 14 days
    last_date = df['timestamp'].max()
    start_date = last_date - pd.Timedelta(days=14)
    
    df_plot = df[df['timestamp'] >= start_date].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df_plot['timestamp'], df_plot['load'], linewidth=1.5)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Load (MW)')
    ax.set_title(f'{country_code} - Last 14 Days of Load Data')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to country-specific folder
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    output_file = country_dir / "last_14_days.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_file}")


def plot_stl_decomposition(
    trend: pd.Series,
    seasonal: pd.Series,
    remainder: pd.Series,
    observed: pd.Series,
    country_code: str,
    output_dir: Path,
    figsize: tuple = (15, 10)
):
    """
    Plot STL decomposition results.
    
    Args:
        trend: Trend component
        seasonal: Seasonal component
        remainder: Remainder component
        observed: Original observed series
        country_code: Country code for title
        output_dir: Base output directory (will create country subfolder)
        figsize: Figure size
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    axes[0].plot(observed.index, observed.values, linewidth=1)
    axes[0].set_ylabel('Observed')
    axes[0].set_title(f'{country_code} - STL Decomposition')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(trend.index, trend.values, linewidth=1, color='orange')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(seasonal.index, seasonal.values, linewidth=1, color='green')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(remainder.index, remainder.values, linewidth=1, color='red')
    axes[3].set_ylabel('Remainder')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to country-specific folder
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    output_file = country_dir / "stl_decomposition.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved STL decomposition plot to {output_file}")


def plot_acf_pacf(
    series: pd.Series,
    country_code: str,
    output_dir: Path,
    lags: int = 48,
    figsize: tuple = (15, 6)
):
    """
    Plot ACF and PACF for a time series.
    
    Args:
        series: Time series to plot
        country_code: Country code for title
        output_dir: Base output directory (will create country subfolder)
        lags: Number of lags to plot
        figsize: Figure size
    """
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f'{country_code} - ACF')
    
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title(f'{country_code} - PACF')
    
    plt.tight_layout()
    
    # Save to country-specific folder
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    output_file = country_dir / "acf_pacf.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ACF/PACF plot to {output_file}")

