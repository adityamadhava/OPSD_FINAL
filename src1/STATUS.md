# src1/ Reorganization Status

## ✅ Completed Files

1. **load_opsd.py** - ✅ Created (from data_preparation.py)
2. **metrics.py** - ✅ Created (metric functions from evaluation.py)
3. **dashboard_app.py** - ✅ Copied (needs import updates)
4. **live_loop.py** - ✅ Copied (needs import updates)
5. **analysis.py** - ✅ Copied (supporting file)
6. **visualization.py** - ✅ Copied (supporting file)

## ⏳ Files To Create

1. **decompose_acf_pacf.py** - Combine STL decomposition + SARIMA grid search
2. **forecast.py** - Combine SARIMA backtesting + Neural models (GRU/LSTM)
3. **anomaly.py** - Z-score + CUSUM detection (from anomaly_detection.py)
4. **anomaly_ml.py** - ML classifier (from anomaly_detection.py)

## 📝 Notes

- All files need import path updates to work with src1/ structure
- Neural models are implemented in src/neural_models.py and need to be integrated into forecast.py
- The original src/ folder remains intact for reference

## Next Steps

1. Create decompose_acf_pacf.py
2. Create forecast.py (with neural model integration)
3. Create anomaly.py
4. Create anomaly_ml.py
5. Update all imports in copied files
6. Test the reorganized structure

