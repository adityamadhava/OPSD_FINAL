"""
Anomaly detection for load forecasting residuals.
Implements residual z-scores and CUSUM as per requirements.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import json
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score

# Try to import LightGBM, fallback to LogisticRegression if not available
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
    print("LightGBM not available, will use LogisticRegression")


def calculate_residuals(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1-step-ahead residuals: et = yt - ŷt
    
    Args:
        forecasts_df: DataFrame with y_true and yhat columns
    
    Returns:
        DataFrame with residuals added
    """
    df = forecasts_df.copy()
    df['residual'] = df['y_true'] - df['yhat']
    return df


def calculate_rolling_zscore(
    df: pd.DataFrame,
    window: int = 336,  # 14 days = 336 hours
    min_periods: int = 168,  # 7 days = 168 hours
    z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Calculate rolling z-score: zt = (et - μroll) / σroll
    
    Args:
        df: DataFrame with residuals
        window: Rolling window in hours (default 336 = 14 days)
        min_periods: Minimum periods for rolling calculation (default 168 = 7 days)
        z_threshold: Threshold for flagging anomalies (default 3.0)
    
    Returns:
        DataFrame with z-scores and flags
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate rolling mean and std
    df['residual_mean_roll'] = df['residual'].rolling(
        window=window, 
        min_periods=min_periods,
        center=False
    ).mean()
    
    df['residual_std_roll'] = df['residual'].rolling(
        window=window,
        min_periods=min_periods,
        center=False
    ).std()
    
    # Calculate z-score
    df['z_resid'] = (df['residual'] - df['residual_mean_roll']) / (df['residual_std_roll'] + 1e-8)
    
    # Flag anomalies: flag_z = 1 if |zt| >= 3.0
    df['flag_z'] = (np.abs(df['z_resid']) >= z_threshold).astype(int)
    
    return df


def calculate_cusum(
    df: pd.DataFrame,
    k: float = 0.5,
    h: float = 5.0
) -> pd.DataFrame:
    """
    Calculate CUSUM on z-scores: k = 0.5, h = 5.0
    
    Args:
        df: DataFrame with z_resid column
        k: CUSUM drift parameter (default 0.5)
        h: CUSUM threshold (default 5.0)
    
    Returns:
        DataFrame with CUSUM values and flags
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    z_scores = df['z_resid'].fillna(0).values
    n = len(z_scores)
    
    # CUSUM calculation on z-scores
    s_plus = np.zeros(n)
    s_minus = np.zeros(n)
    
    for i in range(1, n):
        s_plus[i] = max(0, s_plus[i-1] + z_scores[i] - k)
        s_minus[i] = max(0, s_minus[i-1] - z_scores[i] - k)
    
    df['cusum_plus'] = s_plus
    df['cusum_minus'] = s_minus
    df['cusum_max'] = np.maximum(s_plus, s_minus)
    
    # Flag anomaly if S⁺ > h or S⁻ > h
    df['flag_cusum'] = ((df['cusum_plus'] > h) | (df['cusum_minus'] > h)).astype(int)
    
    return df


def create_silver_labels(
    df: pd.DataFrame,
    z_high: float = 3.5,
    z_medium: float = 2.5,
    z_low: float = 1.0
) -> pd.DataFrame:
    """
    Create silver labels for ML classifier.
    
    Positive if:
    - (|zt| >= 3.5) OR
    - (y_true outside [lo,hi] AND |zt| >= 2.5)
    
    Negative if:
    - |zt| < 1.0 AND y_true inside [lo,hi]
    
    Args:
        df: DataFrame with z_resid, y_true, lo, hi columns
        z_high: High z-score threshold (default 3.5)
        z_medium: Medium z-score threshold (default 2.5)
        z_low: Low z-score threshold (default 1.0)
    
    Returns:
        DataFrame with silver labels
    """
    df = df.copy()
    
    # Check if y_true is outside prediction interval
    if 'lo' in df.columns and 'hi' in df.columns:
        outside_pi = (df['y_true'] < df['lo']) | (df['y_true'] > df['hi'])
        inside_pi = (df['y_true'] >= df['lo']) & (df['y_true'] <= df['hi'])
    else:
        outside_pi = pd.Series([False] * len(df))
        inside_pi = pd.Series([True] * len(df))
    
    abs_z = np.abs(df['z_resid'])
    
    # Positive labels
    positive = (
        (abs_z >= z_high) |
        (outside_pi & (abs_z >= z_medium))
    )
    
    # Negative labels
    negative = (abs_z < z_low) & inside_pi
    
    # Create labels: 1 = positive (anomaly), 0 = negative (normal), -1 = unlabeled
    df['silver_label'] = -1
    df.loc[positive, 'silver_label'] = 1
    df.loc[negative, 'silver_label'] = 0
    
    return df


def create_features_for_classifier(
    df: pd.DataFrame,
    lookback_hours: int = 48
) -> pd.DataFrame:
    """
    Create features from last 24-48h (lags/rollups, calendar, forecast context).
    
    Args:
        df: DataFrame with time series data
        lookback_hours: Number of hours to look back (default 48)
    
    Returns:
        DataFrame with features
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calendar features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # Residual features (lags and rolling stats)
    for lag in [1, 6, 12, 24]:
        if lag <= lookback_hours:
            df[f'residual_lag{lag}'] = df['residual'].shift(lag)
            df[f'z_resid_lag{lag}'] = df['z_resid'].shift(lag)
    
    # Rolling statistics
    for window in [6, 12, 24]:
        if window <= lookback_hours:
            df[f'residual_roll_mean_{window}'] = df['residual'].rolling(window=window, min_periods=1).mean()
            df[f'residual_roll_std_{window}'] = df['residual'].rolling(window=window, min_periods=1).std()
            df[f'z_resid_roll_mean_{window}'] = df['z_resid'].rolling(window=window, min_periods=1).mean()
            df[f'z_resid_roll_max_{window}'] = df['z_resid'].rolling(window=window, min_periods=1).apply(lambda x: np.max(np.abs(x)))
    
    # Load features
    df['load_lag1'] = df['y_true'].shift(1)
    df['load_lag24'] = df['y_true'].shift(24)
    df['load_roll_mean_24'] = df['y_true'].rolling(window=24, min_periods=1).mean()
    
    # Forecast context
    if 'horizon' in df.columns:
        df['forecast_horizon'] = df['horizon']
    else:
        df['forecast_horizon'] = 1
    
    # CUSUM features
    if 'cusum_max' in df.columns:
        df['cusum_max_lag1'] = df['cusum_max'].shift(1)
        df['cusum_max_roll_mean_24'] = df['cusum_max'].rolling(window=24, min_periods=1).mean()
    
    # Fill NaN values
    df = df.bfill().fillna(0)
    
    return df


def sample_for_verification(
    df: pd.DataFrame,
    n_samples: int = 100,
    n_positives: int = 50,
    n_negatives: int = 50,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly sample timestamps for human verification.
    
    Args:
        df: DataFrame with silver labels
        n_samples: Total number of samples
        n_positives: Number of positive samples
        n_negatives: Number of negative samples
        random_state: Random seed
    
    Returns:
        DataFrame with sampled rows for verification
    """
    np.random.seed(random_state)
    
    positives = df[df['silver_label'] == 1].copy()
    negatives = df[df['silver_label'] == 0].copy()
    
    # Sample positives
    if len(positives) >= n_positives:
        pos_sample = positives.sample(n=n_positives, random_state=random_state)
    else:
        pos_sample = positives
    
    # Sample negatives
    if len(negatives) >= n_negatives:
        neg_sample = negatives.sample(n=n_negatives, random_state=random_state)
    else:
        neg_sample = negatives
    
    # Combine and add verification flag
    sample_df = pd.concat([pos_sample, neg_sample], ignore_index=True)
    sample_df['needs_verification'] = True
    
    # Add ±24h context columns for visual check
    sample_df['timestamp_minus_24h'] = sample_df['timestamp'] - pd.Timedelta(hours=24)
    sample_df['timestamp_plus_24h'] = sample_df['timestamp'] + pd.Timedelta(hours=24)
    
    return sample_df


def train_anomaly_classifier(
    df: pd.DataFrame,
    model_type: str = 'lightgbm',
    test_size: float = 0.2,
    random_state: int = 42,
    fixed_precision: float = 0.80
) -> dict:
    """
    Train ML classifier and evaluate at fixed precision.
    
    Args:
        df: DataFrame with features and labels
        model_type: 'logistic' or 'lightgbm'
        test_size: Proportion for test set
        random_state: Random seed
        fixed_precision: Fixed precision for F1 calculation
    
    Returns:
        Dictionary with model and evaluation metrics
    """
    from sklearn.model_selection import train_test_split
    
    # Get labeled data only
    labeled = df[df['silver_label'] >= 0].copy()
    
    if len(labeled) == 0:
        print("  Warning: No labeled data available")
        return None
    
    # Feature columns - exclude non-numeric and label columns
    exclude_cols = [
        'timestamp', 'y_true', 'yhat', 'residual', 'abs_residual',
        'z_resid', 'flag_z', 'flag_cusum', 'silver_label',
        'cusum_plus', 'cusum_minus', 'cusum_max',
        'residual_mean_roll', 'residual_std_roll',
        'needs_verification', 'timestamp_minus_24h', 'timestamp_plus_24h',
        'lo', 'hi', 'horizon', 'train_end'
    ]
    
    # Get numeric columns only
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
    
    X = labeled[feature_cols].values
    y = labeled['silver_label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"  Training samples: {len(X_train)} (positives: {y_train.sum()})")
    print(f"  Test samples: {len(X_test)} (positives: {y_test.sum()})")
    
    # Train model
    if model_type == 'lightgbm' and HAS_LIGHTGBM:
        try:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                class_weight='balanced',
                verbose=-1
            )
            model.fit(X_train, y_train)
        except:
            print("  LightGBM failed, using Logistic Regression")
            model_type = 'logistic'
    else:
        if model_type == 'lightgbm':
            print("  LightGBM not available, using Logistic Regression")
        model_type = 'logistic'
    
    if model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Find threshold for fixed precision
    if len(precision) > 0:
        # Find threshold closest to fixed precision
        precision_idx = np.argmin(np.abs(precision - fixed_precision))
        if precision_idx < len(thresholds):
            fixed_threshold = thresholds[precision_idx]
            y_pred_fixed = (y_pred_proba >= fixed_threshold).astype(int)
            
            f1_fixed = f1_score(y_test, y_pred_fixed)
            precision_fixed = precision_score(y_test, y_pred_fixed)
            recall_fixed = recall_score(y_test, y_pred_fixed)
        else:
            f1_fixed = f1_score(y_test, y_pred)
            precision_fixed = precision_score(y_test, y_pred)
            recall_fixed = recall_score(y_test, y_pred)
    else:
        f1_fixed = f1_score(y_test, y_pred)
        precision_fixed = precision_score(y_test, y_pred)
        recall_fixed = recall_score(y_test, y_pred)
    
    print(f"\n  Evaluation Metrics:")
    print(f"    PR-AUC: {pr_auc:.4f}")
    print(f"    F1 at P={fixed_precision}: {f1_fixed:.4f}")
    print(f"    Precision: {precision_fixed:.4f}")
    print(f"    Recall: {recall_fixed:.4f}")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'pr_auc': pr_auc,
        'f1_at_fixed_precision': f1_fixed,
        'precision': precision_fixed,
        'recall': recall_fixed,
        'fixed_precision': fixed_precision
    }


def detect_anomalies_country(
    country_code: str,
    output_dir: Path,
    split: str = 'dev',
    window: int = 336,
    min_periods: int = 168,
    z_threshold: float = 3.0,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0
) -> pd.DataFrame:
    """
    Detect anomalies for a country's forecasts.
    
    Args:
        country_code: Country code
        output_dir: Output directory
        split: 'dev' or 'test'
        window: Rolling window for z-score (default 336h = 14 days)
        min_periods: Minimum periods for rolling (default 168h = 7 days)
        z_threshold: Z-score threshold (default 3.0)
        cusum_k: CUSUM drift parameter (default 0.5)
        cusum_h: CUSUM threshold (default 5.0)
    
    Returns:
        DataFrame with anomaly detections
    """
    print(f"\n{'='*60}")
    print(f"Anomaly Detection for {country_code} - {split.upper()}")
    print(f"{'='*60}")
    
    # Load forecasts
    forecast_file = output_dir / country_code / f"{country_code}_forecasts_{split}.csv"
    if not forecast_file.exists():
        print(f"  ✗ Forecast file not found: {forecast_file}")
        return None
    
    print(f"\n1. Loading forecasts from {forecast_file}")
    df = pd.read_csv(forecast_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"   Loaded {len(df)} forecast points")
    
    # Calculate residuals
    print("\n2. Calculating 1-step-ahead residuals...")
    df = calculate_residuals(df)
    print(f"   Mean residual: {df['residual'].mean():.2f}")
    print(f"   Std residual: {df['residual'].std():.2f}")
    
    # Calculate rolling z-score
    print(f"\n3. Calculating rolling z-score (window={window}h, min_periods={min_periods}h)...")
    df = calculate_rolling_zscore(df, window=window, min_periods=min_periods, z_threshold=z_threshold)
    n_anomalies_z = df['flag_z'].sum()
    print(f"   Detected {n_anomalies_z} z-score anomalies ({n_anomalies_z/len(df)*100:.2f}%)")
    
    # Calculate CUSUM
    print(f"\n4. Calculating CUSUM (k={cusum_k}, h={cusum_h})...")
    df = calculate_cusum(df, k=cusum_k, h=cusum_h)
    n_anomalies_cusum = df['flag_cusum'].sum()
    print(f"   Detected {n_anomalies_cusum} CUSUM anomalies ({n_anomalies_cusum/len(df)*100:.2f}%)")
    
    # Save basic anomalies file
    print("\n5. Saving anomaly detection results...")
    output_file = output_dir / country_code / f"{country_code}_anomalies.csv"
    
    output_cols = ['timestamp', 'y_true', 'yhat', 'z_resid', 'flag_z']
    if 'flag_cusum' in df.columns:
        output_cols.append('flag_cusum')
    
    df_output = df[output_cols].copy()
    df_output.to_csv(output_file, index=False)
    print(f"   Saved to {output_file}")
    
    return df


def train_ml_classifier_country(
    country_code: str,
    output_dir: Path,
    split: str = 'dev',
    model_type: str = 'lightgbm',
    fixed_precision: float = 0.80
) -> dict:
    """
    Train ML classifier for anomaly detection.
    
    Args:
        country_code: Country code
        output_dir: Output directory
        split: 'dev' or 'test'
        model_type: 'logistic' or 'lightgbm'
        fixed_precision: Fixed precision for F1 calculation
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"ML Classifier Training for {country_code} - {split.upper()}")
    print(f"{'='*60}")
    
    # Load forecasts
    forecast_file = output_dir / country_code / f"{country_code}_forecasts_{split}.csv"
    if not forecast_file.exists():
        print(f"  ✗ Forecast file not found: {forecast_file}")
        return None
    
    df = pd.read_csv(forecast_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate residuals and z-scores (reuse from anomaly detection)
    df = calculate_residuals(df)
    df = calculate_rolling_zscore(df, window=336, min_periods=168, z_threshold=3.0)
    df = calculate_cusum(df, k=0.5, h=5.0)
    
    # Create silver labels
    print("\n1. Creating silver labels...")
    df = create_silver_labels(df)
    
    n_positives = (df['silver_label'] == 1).sum()
    n_negatives = (df['silver_label'] == 0).sum()
    n_unlabeled = (df['silver_label'] == -1).sum()
    
    print(f"   Positives: {n_positives}")
    print(f"   Negatives: {n_negatives}")
    print(f"   Unlabeled: {n_unlabeled}")
    
    # Sample for verification
    print("\n2. Sampling timestamps for human verification...")
    verification_sample = sample_for_verification(df, n_samples=100, n_positives=50, n_negatives=50)
    
    verification_file = output_dir / country_code / f"anomaly_labels_verified_{split}.csv"
    verification_sample.to_csv(verification_file, index=False)
    print(f"   Saved verification sample to {verification_file}")
    print(f"   Please verify these {len(verification_sample)} samples manually")
    
    # Create features
    print("\n3. Creating features for classifier...")
    df = create_features_for_classifier(df, lookback_hours=48)
    
    # Train classifier
    print("\n4. Training classifier...")
    results = train_anomaly_classifier(
        df, 
        model_type=model_type,
        fixed_precision=fixed_precision
    )
    
    if results:
        # Save evaluation results
        eval_file = output_dir / country_code / f"anomaly_ml_eval_{split}.json"
        eval_data = {
            'country': country_code,
            'split': split,
            'pr_auc': float(results['pr_auc']),
            'f1_at_fixed_precision': float(results['f1_at_fixed_precision']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'fixed_precision': float(results['fixed_precision']),
            'model_type': model_type
        }
        
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"\n   Saved evaluation to {eval_file}")
    
    return results


def main():
    """Run anomaly detection for selected country."""
    output_dir = Path(__file__).parent.parent / "outputs"
    
    # =================================================================
    # SELECT COUNTRY TO PROCESS
    # Uncomment the country you want to run
    # =================================================================
    country = 'DE'  # Germany
    # country = 'FR'  # France
    # country = 'ES'  # Spain
    
    # 3.1: Residual z-score + CUSUM
    print(f"\n{'#'*60}")
    print(f"3.1: Residual Z-Score + CUSUM Detection - {country}")
    print(f"{'#'*60}")
    
    anomalies_df = detect_anomalies_country(
        country, output_dir, split='dev',
        window=336,  # 14 days
        min_periods=168,  # 7 days
        z_threshold=3.0,
        cusum_k=0.5,
        cusum_h=5.0
    )
    
    # 3.2: ML-based classifier
    print(f"\n{'#'*60}")
    print(f"3.2: ML-Based Anomaly Classifier - {country}")
    print(f"{'#'*60}")
    
    ml_results = train_ml_classifier_country(
        country, output_dir, split='dev',
        model_type='lightgbm',
        fixed_precision=0.80
    )
    
    print(f"\n{'='*60}")
    print("Anomaly Detection Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
