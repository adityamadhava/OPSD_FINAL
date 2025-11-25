"""
Data preparation module for OPSD time series data.
Extracts and cleans data for specified countries.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def prepare_country_data(
    df: pd.DataFrame,
    country_code: str,
    load_col: Optional[str] = None,
    wind_col: Optional[str] = None,
    solar_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare clean DataFrame for a single country.
    
    Args:
        df: Raw OPSD DataFrame
        country_code: Country code (e.g., 'DE', 'FR', 'ES')
        load_col: Specific load column name (if None, auto-detect)
        wind_col: Specific wind column name (if None, auto-detect)
        solar_col: Specific solar column name (if None, auto-detect)
    
    Returns:
        Cleaned DataFrame with columns: timestamp, load, wind, solar
    """
    # Find columns
    if load_col is None:
        # Try main load column first
        load_candidates = [col for col in df.columns 
                          if f'{country_code}_load_actual' in col 
                          and 'entsoe_transparency' in col]
        if load_candidates:
            load_col = load_candidates[0]
        else:
            # Fallback to any load_actual column
            load_candidates = [col for col in df.columns 
                              if f'{country_code}_load_actual' in col]
            if load_candidates:
                load_col = load_candidates[0]
            else:
                raise ValueError(f"Could not find load column for {country_code}")
    
    if wind_col is None:
        # Try main wind generation column
        wind_candidates = [col for col in df.columns 
                          if f'{country_code}_wind_generation_actual' in col
                          and 'offshore' not in col and 'onshore' not in col]
        if wind_candidates:
            wind_col = wind_candidates[0]
        else:
            # Fallback to onshore wind
            wind_candidates = [col for col in df.columns 
                              if f'{country_code}_wind_onshore_generation_actual' in col]
            if wind_candidates:
                wind_col = wind_candidates[0]
    
    if solar_col is None:
        solar_candidates = [col for col in df.columns 
                           if f'{country_code}_solar_generation_actual' in col]
        if solar_candidates:
            solar_col = solar_candidates[0]
    
    # Create result DataFrame
    result = pd.DataFrame()
    result['timestamp'] = pd.to_datetime(df['utc_timestamp'])
    
    # Extract load
    result['load'] = df[load_col].copy()
    
    # Extract wind (if available)
    if wind_col and wind_col in df.columns:
        result['wind'] = df[wind_col].copy()
    else:
        result['wind'] = np.nan
    
    # Extract solar (if available)
    if solar_col and solar_col in df.columns:
        result['solar'] = df[solar_col].copy()
    else:
        result['solar'] = np.nan
    
    # Drop rows with missing load
    result = result.dropna(subset=['load']).copy()
    
    # Impute missing values in load, wind, solar with mean
    for col in ['load', 'wind', 'solar']:
        if col in result.columns:
            mean_val = result[col].mean()
            if pd.notna(mean_val):
                result[col] = result[col].fillna(mean_val)
            else:
                result[col] = result[col].fillna(0)
    
    # Sort by timestamp
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicate timestamps (keep first)
    result = result.drop_duplicates(subset=['timestamp'], keep='first')
    
    return result


def load_and_prepare_all_countries(
    data_path: Path,
    countries: list = ['DE', 'FR', 'ES']
) -> Dict[str, pd.DataFrame]:
    """
    Load raw data and prepare clean DataFrames for all countries.
    
    Args:
        data_path: Path to raw CSV file
        countries: List of country codes to process
    
    Returns:
        Dictionary mapping country codes to cleaned DataFrames
    """
    print(f"Loading data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    country_data = {}
    
    for country in countries:
        print(f"\nProcessing {country}...")
        try:
            df_country = prepare_country_data(df_raw, country)
            country_data[country] = df_country
            print(f"  ✓ {country}: {len(df_country)} rows, "
                  f"date range: {df_country['timestamp'].min()} to {df_country['timestamp'].max()}")
            print(f"    Load: {df_country['load'].isna().sum()} missing (imputed)")
            print(f"    Wind: {df_country['wind'].isna().sum()} missing (imputed)")
            print(f"    Solar: {df_country['solar'].isna().sum()} missing (imputed)")
        except Exception as e:
            print(f"  ✗ Error processing {country}: {e}")
            raise
    
    return country_data


if __name__ == "__main__":
    # Test the data preparation
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    input_file = data_dir / "time_series_60min.csv"
    
    country_data = load_and_prepare_all_countries(input_file)
    
    # Save processed data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for country, df in country_data.items():
        output_file = processed_dir / f"{country}_clean.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {country} data to {output_file}")

