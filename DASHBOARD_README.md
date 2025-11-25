# Live Load Forecasting Dashboard

## Overview
Streamlit dashboard for monitoring live ingestion and online adaptation simulation.

## Features
- **Country Selector**: Choose between DE, FR, ES (DE preselected as live country)
- **Live Series Chart**: Last 7-14 days of actual vs forecasted load
- **Forecast Cone**: Next 24h forecast with 80% prediction intervals
- **Anomaly Tape**: Highlights hours flagged as anomalies (z-score and CUSUM)
- **KPI Tiles**: 
  - Rolling 7-day MASE
  - 80% PI Coverage (7-day)
  - Anomaly hours today
  - Last update time

## Prerequisites
```bash
# Install dependencies (if not already installed)
source venv/bin/activate
pip install streamlit plotly
```

## Running the Dashboard

```bash
# Activate virtual environment
source venv/bin/activate

# Run dashboard
streamlit run src/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Data Requirements

The dashboard expects the following files in `outputs/{COUNTRY}/`:
- `{COUNTRY}_online_forecasts.csv` - Forecast data from online simulation
- `{COUNTRY}_online_updates.csv` - Update log from online simulation
- `{COUNTRY}_anomalies_dev.csv` or `{COUNTRY}_online_anomalies.csv` - Anomaly detection results

## Generating Required Data

If you haven't run the online adaptation simulation yet:

```bash
# Run online adaptation simulation (generates forecasts and updates)
python src/online_adaptation.py

# Run anomaly detection (if needed)
python src/anomaly_detection.py
```

## Dashboard Layout

- **Top**: KPI tiles showing key metrics
- **Left Column**: 
  - Live series chart (14 days)
  - Forecast cone (next 24h)
- **Right Column**:
  - Anomaly tape visualization
  - Update log summary

## Notes

- The dashboard uses cached data loading for performance
- Data is automatically refreshed when files are updated
- Use the sidebar to switch between countries
- DE is preselected as the "live" country for monitoring

