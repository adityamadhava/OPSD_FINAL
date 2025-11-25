# File Reorganization Summary

This folder (`src1/`) contains files reorganized to match the preferred repository layout from the assignment (Section 6.1).

## File Mapping

| Preferred Name (src1/) | Original Name (src/) | Status |
|------------------------|----------------------|--------|
| `load_opsd.py` | `data_preparation.py` | ✅ Created |
| `metrics.py` | `evaluation.py` (metric functions only) | ✅ Created |
| `decompose_acf_pacf.py` | `stl_decomposition.py` + `sarima_model_selection.py` | ⏳ To create |
| `forecast.py` | `backtest_combined.py` + `neural_models.py` | ⏳ To create |
| `anomaly.py` | `anomaly_detection.py` (z-score + CUSUM parts) | ⏳ To create |
| `anomaly_ml.py` | `anomaly_detection.py` (ML classifier parts) | ⏳ To create |
| `live_loop.py` | `online_adaptation.py` | ✅ Copied (needs import updates) |
| `dashboard_app.py` | `dashboard.py` | ✅ Copied (needs import updates) |

## Supporting Files Needed

- `analysis.py` - STL, stationarity, differencing functions
- `visualization.py` - Plotting utilities

## Next Steps

1. Create `decompose_acf_pacf.py` (combine STL + SARIMA selection)
2. Create `forecast.py` (combine backtesting + neural models)
3. Create `anomaly.py` (z-score + CUSUM only)
4. Create `anomaly_ml.py` (ML classifier only)
5. Update imports in all files to work with new structure
6. Copy supporting files (`analysis.py`, `visualization.py`) to `src1/`

## Import Updates Required

All files need to update imports from:
- `from src.xxx import ...` → `from src1.xxx import ...` or relative imports
- Or use absolute imports from project root

