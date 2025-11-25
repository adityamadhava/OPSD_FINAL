"""
Perform STL decomposition and seasonality analysis for load forecasting.
Analyzes trend, seasonal patterns, and stationarity for each country.
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import perform_stl_decomposition, check_stationarity, suggest_differencing
from src.visualization import plot_stl_decomposition, plot_acf_pacf


def analyze_country_decomposition(
    df: pd.DataFrame,
    country_code: str,
    output_dir: Path
) -> dict:
    """
    Perform comprehensive STL decomposition and seasonality analysis.
    
    Args:
        df: DataFrame with timestamp and load columns
        country_code: Country code
        output_dir: Directory to save plots and results
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {country_code}")
    print(f"{'='*60}")
    
    # Set timestamp as index for analysis
    df_analysis = df.set_index('timestamp').copy()
    series = df_analysis['load']
    
    # STL Decomposition
    print("\n1. Performing STL decomposition (period=24 for daily seasonality)...")
    trend, seasonal, remainder = perform_stl_decomposition(series, period=24, robust=True)
    
    # Save STL plot
    print("   Saving STL decomposition plot...")
    plot_stl_decomposition(trend, seasonal, remainder, series, country_code, output_dir)
    
    # Stationarity tests
    print("\n2. Checking stationarity...")
    original_stat = check_stationarity(series)
    print(f"   Original series:")
    print(f"     ADF statistic: {original_stat['adf_statistic']:.4f}")
    print(f"     p-value: {original_stat['p_value']:.6f}")
    print(f"     Stationary: {original_stat['is_stationary']}")
    print(f"     Critical values: {original_stat['critical_values']}")
    
    # Suggest differencing
    print("\n3. Suggesting differencing orders...")
    d, D = suggest_differencing(series, seasonal_period=24)
    print(f"   Regular differencing (d): {d}")
    print(f"   Seasonal differencing (D): {D}")
    
    # Prepare differenced series for ACF/PACF
    series_diff = series.copy()
    if d > 0:
        series_diff = series_diff.diff(d).dropna()
        print(f"   Applied {d} regular difference(s)")
    if D > 0:
        series_diff = series_diff.diff(24).dropna()
        print(f"   Applied {D} seasonal difference(s) with period 24")
    
    # Check stationarity of differenced series
    diff_stat = None
    if len(series_diff) > 0:
        diff_stat = check_stationarity(series_diff)
        print(f"\n   Differenced series:")
        print(f"     ADF statistic: {diff_stat['adf_statistic']:.4f}")
        print(f"     p-value: {diff_stat['p_value']:.6f}")
        print(f"     Stationary: {diff_stat['is_stationary']}")
    
    # ACF/PACF analysis
    print("\n4. Computing ACF/PACF (up to lag 48)...")
    if len(series_diff) > 48:
        plot_acf_pacf(series_diff, country_code, output_dir, lags=48)
    else:
        print("   Skipping ACF/PACF - insufficient data after differencing")
    
    # Summary statistics
    print("\n5. Summary statistics:")
    print(f"   Original series - Mean: {series.mean():.2f}, Std: {series.std():.2f}")
    print(f"   Trend - Mean: {trend.mean():.2f}, Std: {trend.std():.2f}")
    print(f"   Seasonal - Mean: {seasonal.mean():.2f}, Std: {seasonal.std():.2f}")
    print(f"   Remainder - Mean: {remainder.mean():.2f}, Std: {remainder.std():.2f}")
    
    return {
        'trend': trend,
        'seasonal': seasonal,
        'remainder': remainder,
        'original_stationarity': original_stat,
        'differenced_stationarity': diff_stat if len(series_diff) > 0 else None,
        'd': d,
        'D': D,
        'differenced_series': series_diff
    }


def main():
    """Perform STL decomposition for all countries."""
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent / "outputs"
    
    countries = ['DE', 'FR', 'ES']
    results = {}
    
    for country in countries:
        input_file = processed_dir / f"{country}_clean.csv"
        
        if not input_file.exists():
            print(f"✗ File not found: {input_file}")
            continue
        
        # Load data
        df = pd.read_csv(input_file)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Perform analysis
        result = analyze_country_decomposition(df, country, output_dir)
        results[country] = result
    
    print(f"\n{'='*60}")
    print("STL Decomposition Analysis Complete")
    print(f"{'='*60}")
    print(f"\nAll plots saved to country-specific folders in: {output_dir}")
    
    # Print summary table
    print("\nSummary of Differencing Recommendations:")
    print("-" * 60)
    print(f"{'Country':<10} {'d':<5} {'D':<5} {'Original Stationary':<20}")
    print("-" * 60)
    for country, result in results.items():
        is_stat = result['original_stationarity']['is_stationary']
        print(f"{country:<10} {result['d']:<5} {result['D']:<5} {str(is_stat):<20}")
    print("-" * 60)


if __name__ == "__main__":
    main()

