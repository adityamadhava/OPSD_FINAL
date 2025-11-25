# OPSD Electric Load Forecasting System

A comprehensive day-ahead (24-step) electric load forecasting system using Open Power System Data (OPSD) hourly time series data. This system includes SARIMA modeling, anomaly detection, online adaptation, and a live monitoring dashboard.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
- [Output Files](#output-files)
- [Running the Dashboard](#running-the-dashboard)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Setup

### 1. Clone or Download the Repository

```bash
# If using git
git clone https://github.com/adityamadhava/OPSD_FINAL.git
cd OPSD_FINAL

# Or extract the downloaded zip file and navigate to the directory
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with `lightgbm` on macOS, the code will automatically fall back to Logistic Regression for anomaly detection.

### 4. Verify Data File

Ensure the raw data file exists:
```
data/raw/time_series_60min.csv
```

If the file is missing, download it from the OPSD website or place it in the `data/raw/` directory.

## Project Structure

```
OPSD_Final_TEST_V1/
├── data/
│   ├── raw/
│   │   └── time_series_60min.csv      # Raw OPSD data
│   └── processed/
│       ├── DE_clean.csv               # Cleaned data per country
│       ├── FR_clean.csv
│       └── ES_clean.csv
├── src/
│   ├── data_preparation.py            # Step 1: Data cleaning and preparation
│   ├── create_sanity_plots.py         # Step 2: Basic sanity plots
│   ├── stl_decomposition.py          # Step 3: STL decomposition and seasonality
│   ├── sarima_model_selection.py      # Step 4: SARIMA model selection
│   ├── data_splits.py                 # Step 5: Train/dev/test splits
│   ├── backtest_combined.py           # Step 6: Backtesting framework
│   ├── evaluation.py                  # Step 7: Evaluation metrics
│   ├── anomaly_detection.py           # Step 8: Anomaly detection
│   ├── visualize_anomalies.py         # Step 9: Anomaly visualization
│   ├── online_adaptation.py           # Step 10: Live ingestion simulation
│   ├── dashboard.py                   # Step 11: Streamlit dashboard
│   └── ...                            # Supporting modules
├── outputs/
│   ├── DE/                            # Germany outputs
│   ├── FR/                            # France outputs
│   └── ES/                            # Spain outputs
├── requirements.txt
└── README.md
```

## Step-by-Step Execution Guide

**Important:** Run these steps **sequentially** and **one country at a time**. Before running each script, edit it to uncomment the country you want to process.

### Step 1: Data Preparation

Clean and prepare data for selected countries.

```bash
# Edit src/data_preparation.py and uncomment your country (DE, FR, or ES)
python src/data_preparation.py
```

**What it does:**
- Filters data to last 3 years
- Extracts load, wind, and solar generation data
- Handles missing values (imputes with mean)
- Saves cleaned data to `data/processed/{COUNTRY}_clean.csv`

**Output:** `data/processed/{COUNTRY}_clean.csv`

---

### Step 2: Basic Sanity Plots

Generate plots for the last 14 days to verify data quality.

```bash
# Edit src/create_sanity_plots.py and uncomment your country
python src/create_sanity_plots.py
```

**What it does:**
- Plots last 14 days of load data
- Validates hourly cadence and realistic magnitudes

**Output:** `outputs/{COUNTRY}/last_14_days.png`

---

### Step 3: STL Decomposition and Seasonality Analysis

Perform STL decomposition and analyze seasonality patterns.

```bash
# Edit src/stl_decomposition.py and uncomment your country
python src/stl_decomposition.py
```

**What it does:**
- Performs STL decomposition (period=24 for daily seasonality)
- Checks stationarity using ADF test
- Suggests differencing orders (d, D)
- Generates ACF/PACF plots

**Outputs:**
- `outputs/{COUNTRY}/stl_decomposition.png`
- `outputs/{COUNTRY}/acf_pacf.png`

---

### Step 4: SARIMA Model Selection

Perform AIC/BIC grid search to select optimal SARIMA model.

```bash
# Edit src/sarima_model_selection.py and uncomment your country
python src/sarima_model_selection.py
```

**What it does:**
- Grid search over p,q,P,Q ∈ [0,1] with d=1, D=1, s=24
- Selects model with lowest BIC (ties broken with AIC)
- **Estimated time:** 5-10 minutes per country

**Output:** `outputs/{COUNTRY}/sarima_grid_search.csv`

---

### Step 5: Create Train/Dev/Test Splits

Split data chronologically into 80% train, 10% dev, 10% test.

```bash
# Edit src/data_splits.py and uncomment your country
python src/data_splits.py
```

**What it does:**
- Creates chronological splits (80/10/10)
- Saves train, dev, and test CSV files

**Outputs:**
- `outputs/{COUNTRY}/train.csv`
- `outputs/{COUNTRY}/dev.csv`
- `outputs/{COUNTRY}/test.csv`

---

### Step 6: Backtesting

Run expanding-origin backtesting on dev and test sets.

```bash
# Edit src/backtest_combined.py and uncomment your country
python src/backtest_combined.py
```

**What it does:**
- Expanding-origin backtesting with stride=168h (weekly), horizon=168h
- Generates forecasts with 80% prediction intervals
- **Estimated time:** 30-60 minutes per country (depends on data size)

**Outputs:**
- `outputs/{COUNTRY}/{COUNTRY}_forecasts_dev.csv`
- `outputs/{COUNTRY}/{COUNTRY}_forecasts_test.csv`

**Note:** This step is memory-intensive. If your system hangs, try running for one country at a time.

---

### Step 7: Evaluation Metrics

Calculate MASE, sMAPE, MSE, RMSE, MAPE, and 80% PI coverage.

```bash
# Edit src/evaluation.py and uncomment your country
python src/evaluation.py
```

**What it does:**
- Computes all evaluation metrics
- Creates summary table

**Output:** `outputs/evaluation_results.csv`

---

### Step 8: Anomaly Detection

Detect anomalies using residual z-scores, CUSUM, and ML classifier.

```bash
# Edit src/anomaly_detection.py and uncomment your country
python src/anomaly_detection.py
```

**What it does:**
- Computes 1-step-ahead residuals
- Rolling z-score (window=336h, threshold=3.0)
- CUSUM detection (k=0.5, h=5.0)
- ML-based classifier (Logistic/LightGBM) with silver labels
- **Estimated time:** 10-15 minutes per country

**Outputs:**
- `outputs/{COUNTRY}/{COUNTRY}_anomalies_dev.csv`
- `outputs/{COUNTRY}/anomaly_labels_verified_dev.csv`
- `outputs/{COUNTRY}/anomaly_ml_eval_dev.json`

---

### Step 9: Visualize Anomalies

Create visualization plots for detected anomalies.

```bash
# Edit src/visualize_anomalies.py and uncomment your country
python src/visualize_anomalies.py
```

**What it does:**
- Creates 5-panel plot: time series, residuals, z-scores, CUSUM, ML predictions
- Highlights detected anomalies

**Output:** `outputs/{COUNTRY}/anomalies_visualization_dev.png`

---

### Step 10: Online Adaptation Simulation

Simulate live data ingestion and online model adaptation.

```bash
# Edit src/online_adaptation.py and uncomment your country
python src/online_adaptation.py
```

**What it does:**
- Simulates hourly data ingestion
- Forecasts next 24h at 00:00 UTC
- Detects drift using EWMA of |z|
- Triggers rolling SARIMA refit (last 90 days) on drift or scheduled updates
- **Note:** Currently set to 100 hours for testing. Change `start_hours` parameter for full 2000+ hours.

**Outputs:**
- `outputs/{COUNTRY}/{COUNTRY}_online_forecasts.csv`
- `outputs/{COUNTRY}/{COUNTRY}_online_updates.csv`

---

### Step 11: Dashboard (Optional)

Launch Streamlit dashboard for live monitoring.

```bash
# Make sure you've run online_adaptation.py first
streamlit run src/dashboard.py
```

**What it does:**
- Displays live series (last 7-14 days)
- Forecast cone (next 24h with 80% PI)
- Anomaly tape
- KPI tiles (rolling-7d MASE, PI coverage, anomaly count)

**Access:** Open browser to `http://localhost:8501`

**To stop:** Press `Ctrl+C` in terminal, or run:
```bash
lsof -ti:8501 | xargs kill -9
```

---

## Output Files

### Per-Country Outputs (`outputs/{COUNTRY}/`)

- `last_14_days.png` - Sanity plot
- `stl_decomposition.png` - STL decomposition visualization
- `acf_pacf.png` - ACF/PACF plots
- `sarima_grid_search.csv` - All SARIMA models tested
- `train.csv`, `dev.csv`, `test.csv` - Data splits
- `{COUNTRY}_forecasts_dev.csv` - Dev set forecasts
- `{COUNTRY}_forecasts_test.csv` - Test set forecasts
- `{COUNTRY}_anomalies_dev.csv` - Anomaly detection results
- `anomaly_labels_verified_dev.csv` - Verified anomaly labels
- `anomaly_ml_eval_dev.json` - ML classifier evaluation
- `anomalies_visualization_dev.png` - Anomaly visualization
- `{COUNTRY}_online_forecasts.csv` - Online simulation forecasts
- `{COUNTRY}_online_updates.csv` - Online adaptation log

### Global Outputs (`outputs/`)

- `evaluation_results.csv` - Summary of all evaluation metrics

---

## Running the Dashboard

### Prerequisites

1. Complete Steps 1-10 for at least one country (DE recommended)
2. Ensure `{COUNTRY}_online_forecasts.csv` and `{COUNTRY}_online_updates.csv` exist

### Start Dashboard

```bash
# Activate virtual environment
source venv/bin/activate

# Run dashboard
streamlit run src/dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

### Stop Dashboard

- Press `Ctrl+C` in the terminal, or
- Run: `lsof -ti:8501 | xargs kill -9`

---

## Troubleshooting

### Issue: `ModuleNotFoundError`

**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: `FileNotFoundError` for data files

**Solution:** Ensure you've run previous steps in order. Each step depends on outputs from previous steps.

### Issue: System hangs during backtesting

**Solution:** 
- Run one country at a time
- Close other applications to free memory
- The script uses memory-efficient processing, but large datasets can still be resource-intensive

### Issue: LightGBM installation fails (macOS)

**Solution:** The code automatically falls back to Logistic Regression. This is handled in `anomaly_detection.py`.

### Issue: Streamlit dashboard shows "nothing"

**Solution:**
- Ensure you've run `online_adaptation.py` first
- Check that `{COUNTRY}_online_forecasts.csv` exists
- Verify the country code in `dashboard.py` matches your processed country

### Issue: Port 8501 already in use

**Solution:**
```bash
lsof -ti:8501 | xargs kill -9
```

---

## Notes

- **Run one country at a time:** Edit each script to uncomment the country you want to process
- **Sequential execution:** Each step depends on outputs from previous steps
- **Memory usage:** Backtesting is memory-intensive; close other applications if needed
- **Time estimates:** Total time for one country: ~2-3 hours (depending on hardware)

---

## Support

For issues or questions, please check:
1. Error messages in terminal output
2. Output files in `outputs/` directory
3. This README's troubleshooting section

---

## License

This project is for educational and research purposes.

