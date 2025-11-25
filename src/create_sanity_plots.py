# Create basic sanity plots for the last 14 days of load data per country.
# Validates hourly cadence and realistic magnitudes.
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import visualization module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import plot_last_14_days


def main():
    """Create sanity plots for all countries."""
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent / "outputs"
    
    countries = ['DE', 'FR', 'ES']
    
    for country in countries:
        print(f"\nProcessing {country}...")
        input_file = processed_dir / f"{country}_clean.csv"
        
        if not input_file.exists():
            print(f"  ✗ File not found: {input_file}")
            continue
        
        # Load data
        df = pd.read_csv(input_file)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create plot
        plot_last_14_days(df, country, output_dir)
    
    print(f"\n✓ All sanity plots saved to country-specific folders in {output_dir}")


if __name__ == "__main__":
    main()

