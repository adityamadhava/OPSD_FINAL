"""
Time series analysis: STL decomposition, stationarity, ACF/PACF.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


def perform_stl_decomposition(
    series: pd.Series,
    period: int = 24,
    robust: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Perform STL decomposition on a time series.
    
    Args:
        series: Time series to decompose
        period: Seasonal period (24 for hourly daily seasonality)
        robust: Use robust decomposition
    
    Returns:
        Tuple of (trend, seasonal, remainder)
    """
    stl = STL(series, period=period, robust=robust)
    result = stl.fit()
    
    return result.trend, result.seasonal, result.resid


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict:
    """
    Check stationarity using Augmented Dickey-Fuller test.
    
    Args:
        series: Time series to test
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    result = adfuller(series.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < alpha,
        'critical_values': result[4]
    }


def suggest_differencing(
    series: pd.Series,
    seasonal_period: int = 24
) -> Tuple[int, int]:
    """
    Suggest differencing orders based on stationarity tests.
    
    Args:
        series: Time series to analyze
        seasonal_period: Seasonal period for seasonal differencing
    
    Returns:
        Tuple of (d, D) where d is regular differencing and D is seasonal differencing
    """
    d = 0
    D = 0
    
    # Check original series
    stat_result = check_stationarity(series)
    if not stat_result['is_stationary']:
        # Try first difference
        diff1 = series.diff().dropna()
        stat_result = check_stationarity(diff1)
        if not stat_result['is_stationary']:
            d = 1
        else:
            d = 1
    else:
        d = 0
    
    # Check for seasonal differencing
    if len(series) > seasonal_period * 2:
        seasonal_diff = series.diff(seasonal_period).dropna()
        stat_result = check_stationarity(seasonal_diff)
        if not stat_result['is_stationary']:
            D = 1
    
    return d, D


def analyze_series(
    df: pd.DataFrame,
    country_code: str,
    output_dir: Path
) -> Dict:
    """
    Perform comprehensive analysis on a country's load series.
    
    Args:
        df: DataFrame with timestamp and load columns
        country_code: Country code
        output_dir: Directory to save plots
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\nAnalyzing {country_code}...")
    
    # Set timestamp as index for analysis
    df_analysis = df.set_index('timestamp').copy()
    series = df_analysis['load']
    
    # STL Decomposition
    print("  Performing STL decomposition...")
    trend, seasonal, remainder = perform_stl_decomposition(series, period=24)
    
    # Stationarity tests
    print("  Checking stationarity...")
    original_stat = check_stationarity(series)
    print(f"    Original series - ADF p-value: {original_stat['p_value']:.4f}, "
          f"Stationary: {original_stat['is_stationary']}")
    
    # Suggest differencing
    d, D = suggest_differencing(series, seasonal_period=24)
    print(f"  Suggested differencing: d={d}, D={D}")
    
    # Prepare differenced series for ACF/PACF
    series_diff = series.copy()
    if d > 0:
        series_diff = series_diff.diff(d).dropna()
    if D > 0:
        series_diff = series_diff.diff(24).dropna()
    
    return {
        'trend': trend,
        'seasonal': seasonal,
        'remainder': remainder,
        'original_stationarity': original_stat,
        'd': d,
        'D': D,
        'differenced_series': series_diff
    }

