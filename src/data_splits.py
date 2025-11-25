"""
Create chronological train/dev/test splits (80/10/10) for time series forecasting.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> tuple:
    """
    Split time series data chronologically into train/dev/test sets.
    
    Args:
        df: DataFrame with timestamp index, sorted chronologically
        train_ratio: Proportion for training set (default 0.8)
        dev_ratio: Proportion for dev/validation set (default 0.1)
        test_ratio: Proportion for test set (default 0.1)
    
    Returns:
        Tuple of (train_df, dev_df, test_df)
    """
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    
    train_df = df.iloc[:train_end].copy()
    dev_df = df.iloc[train_end:dev_end].copy()
    test_df = df.iloc[dev_end:].copy()
    
    return train_df, dev_df, test_df


def split_country_data(
    country_code: str,
    processed_dir: Path,
    output_dir: Path
) -> dict:
    """
    Load and split data for a single country.
    
    Args:
        country_code: Country code (DE, FR, ES)
        processed_dir: Directory with cleaned CSV files
        output_dir: Directory to save split data
    
    Returns:
        Dictionary with split information
    """
    print(f"\n{'='*60}")
    print(f"Creating Data Splits for {country_code}")
    print(f"{'='*60}")
    
    # Load cleaned data
    input_file = processed_dir / f"{country_code}_clean.csv"
    if not input_file.exists():
        print(f"  ✗ File not found: {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n  Total data points: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Create splits
    train_df, dev_df, test_df = create_splits(df, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
    
    # Print split information
    print(f"\n  Split sizes:")
    print(f"    Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"      {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"    Dev:   {len(dev_df)} ({len(dev_df)/len(df)*100:.1f}%)")
    print(f"      {dev_df['timestamp'].min()} to {dev_df['timestamp'].max()}")
    print(f"    Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"      {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # Save splits
    country_output_dir = output_dir / country_code
    country_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = country_output_dir / "train.csv"
    dev_file = country_output_dir / "dev.csv"
    test_file = country_output_dir / "test.csv"
    
    train_df.to_csv(train_file, index=False)
    dev_df.to_csv(dev_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n  Saved splits to:")
    print(f"    {train_file}")
    print(f"    {dev_file}")
    print(f"    {test_file}")
    
    return {
        'train': train_df,
        'dev': dev_df,
        'test': test_df,
        'train_file': train_file,
        'dev_file': dev_file,
        'test_file': test_file
    }


def main():
    """Create train/dev/test splits for all countries."""
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
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
    all_splits = {}
    
    for country in countries:
        splits = split_country_data(country, processed_dir, output_dir)
        if splits:
            all_splits[country] = splits
    
    # Print summary
    print(f"\n{'='*60}")
    print("Data Splits Summary")
    print(f"{'='*60}")
    print(f"\n{'Country':<10} {'Train':<12} {'Dev':<12} {'Test':<12} {'Total':<12}")
    print("-" * 60)
    
    for country, splits in all_splits.items():
        train_n = len(splits['train'])
        dev_n = len(splits['dev'])
        test_n = len(splits['test'])
        total_n = train_n + dev_n + test_n
        print(f"{country:<10} {train_n:<12} {dev_n:<12} {test_n:<12} {total_n:<12}")
    
    print("-" * 60)
    print(f"\nAll splits saved to country-specific folders in: {output_dir}")


if __name__ == "__main__":
    main()

