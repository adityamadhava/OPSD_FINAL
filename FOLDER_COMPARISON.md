# Folder Comparison: src/ vs src1/

## ✅ Confirmation

### src/ folder - COMPLETE functionality, but naming doesn't match assignment

**All functionality is implemented:**
- ✅ `data_preparation.py` - Data loading and cleaning
- ✅ `create_sanity_plots.py` - Basic sanity plots
- ✅ `stl_decomposition.py` - STL decomposition
- ✅ `sarima_model_selection.py` - SARIMA grid search
- ✅ `data_splits.py` - Train/dev/test splits
- ✅ `backtest_combined.py` - Backtesting framework
- ✅ `evaluation.py` - Evaluation metrics
- ✅ `anomaly_detection.py` - Anomaly detection (z-score + CUSUM + ML)
- ✅ `visualize_anomalies.py` - Anomaly visualization
- ✅ `online_adaptation.py` - Live ingestion simulation
- ✅ `dashboard.py` - Streamlit dashboard
- ✅ `neural_models.py` - GRU/LSTM models (NEW - just added)
- ✅ `analysis.py` - Supporting functions
- ✅ `visualization.py` - Supporting functions

**Issue:** File names don't match the preferred repository layout from assignment Section 6.1

---

### src1/ folder - Matches preferred naming, but incomplete

**Files created (matching preferred layout):**
- ✅ `load_opsd.py` - Renamed from `data_preparation.py`
- ✅ `metrics.py` - Extracted from `evaluation.py`
- ✅ `dashboard_app.py` - Copied from `dashboard.py`
- ✅ `live_loop.py` - Copied from `online_adaptation.py`
- ✅ `analysis.py` - Copied (supporting)
- ✅ `visualization.py` - Copied (supporting)

**Files still missing:**
- ❌ `decompose_acf_pacf.py` - Should combine STL + SARIMA selection
- ❌ `forecast.py` - Should combine SARIMA backtesting + Neural models
- ❌ `anomaly.py` - Should extract z-score + CUSUM from anomaly_detection.py
- ❌ `anomaly_ml.py` - Should extract ML classifier from anomaly_detection.py

**Status:** 
- ✅ Naming matches assignment preferred layout
- ❌ Functionality is incomplete (only 6/10 files created)
- ❌ Imports not updated yet
- ❌ Nothing new - just reorganization/renaming

---

## Assignment Preferred Layout (Section 6.1)

| Preferred Name | Current src/ Name | Status in src1/ |
|----------------|------------------|-----------------|
| `load_opsd.py` | `data_preparation.py` | ✅ Created |
| `decompose_acf_pacf.py` | `stl_decomposition.py` + `sarima_model_selection.py` | ❌ Missing |
| `forecast.py` | `backtest_combined.py` + `neural_models.py` | ❌ Missing |
| `anomaly.py` | `anomaly_detection.py` (z-score + CUSUM parts) | ❌ Missing |
| `anomaly_ml.py` | `anomaly_detection.py` (ML classifier parts) | ❌ Missing |
| `live_loop.py` | `online_adaptation.py` | ✅ Copied |
| `dashboard_app.py` | `dashboard.py` | ✅ Copied |
| `metrics.py` | `evaluation.py` (metric functions) | ✅ Created |

---

## Summary

**Your understanding is CORRECT:**

1. ✅ **src/ folder is COMPLETE** - All functionality implemented, but naming doesn't match assignment
2. ✅ **src1/ folder matches naming** - But only 6/10 files created, nothing new (just reorganization)

**What's needed:**
- Complete the remaining 4 files in src1/
- Update imports
- Test the reorganized structure

