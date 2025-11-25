"""
Script to clean the time series data by keeping only the last 3 years.
"""
import pandas as pd
from pathlib import Path

# Define paths
data_dir = Path(__file__).parent.parent / "data" / "raw"
input_file = data_dir / "time_series_60min.csv"
output_file = data_dir / "time_series_60min.csv"  # Overwrite original

print(f"Reading data from {input_file}")
df = pd.read_csv(input_file)

# Parse the UTC timestamp column
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Get the maximum date in the dataset
max_date = df['utc_timestamp'].max()
print(f"Maximum date in dataset: {max_date}")

# Calculate the cutoff date (3 years before the max date)
cutoff_date = max_date - pd.DateOffset(years=3)
print(f"Keeping data from {cutoff_date} to {max_date}")

# Filter the dataframe
df_filtered = df[df['utc_timestamp'] >= cutoff_date].copy()

print(f"Original data: {len(df)} rows")
print(f"Filtered data: {len(df_filtered)} rows")
print(f"Removed {len(df) - len(df_filtered)} rows ({((len(df) - len(df_filtered)) / len(df) * 100):.1f}%)")

# Convert timestamp back to string format for CSV
df_filtered['utc_timestamp'] = df_filtered['utc_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
print(f"Saving cleaned data to {output_file}")
df_filtered.to_csv(output_file, index=False)


print("Data cleaning completed successfully!")

