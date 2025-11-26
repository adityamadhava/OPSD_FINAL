"""
Unified Backtesting Script
Runs SARIMA expanding-origin backtesting for DEV and TEST splits
in a single workflow.

Memory-efficient, supports weekly forecasting, and handles
train/dev/test data independently.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import gc
from typing import Tuple

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from statsmodels.tsa.statespace.sarimax import SARIMAX



def load_model_params(country_code: str, output_dir: Path) -> dict:
    grid_file = output_dir / country_code / "sarima_grid_search.csv"
    if not grid_file.exists():
        raise FileNotFoundError(f"Grid search results not found: {grid_file}")

    df = pd.read_csv(grid_file)
    best = df.iloc[0]  # Best model (lowest BIC)

    return {
        "p": int(best["p"]),
        "d": int(best["d"]),
        "q": int(best["q"]),
        "P": int(best["P"]),
        "D": int(best["D"]),
        "Q": int(best["Q"]),
        "s": int(best["s"]),
    }


# =====================================================================
#                    FIT + FORECAST (MEMORY EFFICIENT)
# =====================================================================
def fit_and_forecast(
    train_data: pd.Series,
    model_params: dict,
    horizon: int = 24,
    return_conf_int: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient SARIMA forecast.
    """
    try:
        model = SARIMAX(
            train_data,
            order=(model_params["p"], model_params["d"], model_params["q"]),
            seasonal_order=(
                model_params["P"],
                model_params["D"],
                model_params["Q"],
                model_params["s"],
            ),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted = model.fit(disp=False, maxiter=50)

        if return_conf_int:
            forecast = fitted.get_forecast(steps=horizon)
            yhat = forecast.predicted_mean.values
            conf = forecast.conf_int()
            lo = conf.iloc[:, 0].values
            hi = conf.iloc[:, 1].values
        else:
            forecast = fitted.forecast(steps=horizon)
            yhat = forecast.values
            lo = hi = np.full(horizon, np.nan)

        # Cleanup
        del model, fitted, forecast
        gc.collect()

        return yhat, lo, hi

    except Exception as e:
        print(f"    Warning: Forecast failed: {e}")
        return (
            np.full(horizon, np.nan),
            np.full(horizon, np.nan),
            np.full(horizon, np.nan),
        )


# =====================================================================
#                     BACKTESTING CORE (EXPANDING ORIGIN)
# =====================================================================
def backtest_expanding_origin(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_params: dict,
    horizon: int,
    stride: int,
    warmup_days: int,
) -> pd.DataFrame:

    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    train_series = train_df.set_index("timestamp")["load"]
    test_series = test_df.set_index("timestamp")["load"]

    min_train_size = warmup_days * 24

    timestamps = test_df["timestamp"].values

    # All origin points sampled every "stride" hours
    origins = list(range(0, len(timestamps) - horizon + 1, stride))

    print(f"    Generating {len(origins)} forecasts... (H={horizon}, S={stride})")

    results = []

    for idx, origin_idx in enumerate(origins):
        if (idx + 1) % 10 == 0:
            print(f"      Progress {idx + 1}/{len(origins)}", end="\r")

        # Origin timestamp
        origin_time = timestamps[origin_idx]

        # Create training pool = train + test up to origin
        test_up_to_origin = test_series.iloc[:origin_idx]
        if len(test_up_to_origin) == 0:
            combined_train = train_series
        else:
            combined_train = pd.concat([train_series, test_up_to_origin])

        if len(combined_train) < min_train_size:
            continue

        if origin_idx + horizon > len(test_series):
            continue

        y_true = test_series.iloc[origin_idx : origin_idx + horizon].values
        true_ts = timestamps[origin_idx : origin_idx + horizon]

        yhat, lo, hi = fit_and_forecast(combined_train, model_params, horizon=horizon)

        for h in range(horizon):
            results.append(
                {
                    "timestamp": true_ts[h],
                    "y_true": y_true[h],
                    "yhat": yhat[h],
                    "lo": lo[h],
                    "hi": hi[h],
                    "horizon": h + 1,
                    "train_end": origin_time,
                }
            )

        del combined_train, y_true, yhat, lo, hi
        if idx % 20 == 0:
            gc.collect()

    print(f"\n    Completed {len(origins)} forecasts")
    return pd.DataFrame(results)


# =====================================================================
#                    RUN BACKTEST FOR A COUNTRY + SPLIT
# =====================================================================
def backtest_country(
    country_code: str,
    output_dir: Path,
    split: str,
    horizon: int,
    stride: int,
    warmup_days: int = 60,
) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print(f"Backtesting {country_code} - {split.upper()}")
    print(f"{'='*60}")

    # Load model parameters
    print("\n1. Loading model parameters...")
    params = load_model_params(country_code, output_dir)
    print(
        f"   Model: SARIMA({params['p']},{params['d']},{params['q']})"
        f"({params['P']},{params['D']},{params['Q']}){params['s']}"
    )

    # Load train + split data
    print(f"\n2. Loading {split} data...")
    train_file = output_dir / country_code / "train.csv"
    split_file = output_dir / country_code / f"{split}.csv"

    train_df = pd.read_csv(train_file)
    train_df.columns = train_df.columns.str.strip()
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])

    test_df = pd.read_csv(split_file)
    test_df.columns = test_df.columns.str.strip()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    print(f"   Train: {len(train_df)} rows")
    print(f"   {split.capitalize()}: {len(test_df)} rows")

    print(f"\n3. Running backtest...")
    forecasts = backtest_expanding_origin(
        train_df,
        test_df,
        params,
        horizon=horizon,
        stride=stride,
        warmup_days=warmup_days,
    )

    print(f"\n4. Completed → {len(forecasts)} forecast rows")
    return forecasts


# =====================================================================
#                    MAIN: RUN DEV + TEST FOR ALL COUNTRIES
# =====================================================================
def main():
    """Run backtesting for all countries (dev + test sets).
    
    Comment/uncomment countries below to run specific ones.
    """
    output_dir = Path(__file__).parent.parent / "outputs"
    
    # =================================================================
    # SELECT COUNTRIES TO PROCESS
    # Uncomment the countries you want to run
    # =================================================================
    countries = [
        'DE',  # Germany
        'FR',  # France
        'ES',  # Spain
    ]
    
    # Process each country
    for country in countries:
        try:
            print("\n" + "=" * 60)
            print(f"PROCESSING COUNTRY: {country}")
            print("=" * 60)
            
            # -----------------------------------------------------------------
            # DEV SET
            # -----------------------------------------------------------------
            print("\n" + "#" * 60)
            print(f"Processing {country} DEV (Weekly forecasts)")
            print("#" * 60)

            dev_forecasts = backtest_country(
                country,
                output_dir,
                split="dev",
                horizon=24,
                stride=24,
                warmup_days=60,
            )

            dev_path = output_dir / country / f"{country}_forecasts_dev.csv"
            dev_forecasts.to_csv(dev_path, index=False)
            print(f"Saved DEV forecasts → {dev_path}")

            del dev_forecasts
            gc.collect()

            # -----------------------------------------------------------------
            # TEST SET
            # -----------------------------------------------------------------
            print("\n" + "#" * 60)
            print(f"Processing {country} TEST (Weekly forecasts)")
            print("#" * 60)

            test_forecasts = backtest_country(
                country,
                output_dir,
                split="test",
                horizon=168,
                stride=168,
                warmup_days=60,
            )

            test_path = output_dir / country / f"{country}_forecasts_test.csv"
            test_forecasts.to_csv(test_path, index=False)
            print(f"Saved TEST forecasts → {test_path}")

            del test_forecasts
            gc.collect()
            
            print(f"\n✓ Completed {country}")
            
        except Exception as e:
            print(f"\n✗ Error processing {country}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("Backtesting Complete - All Countries Processed")
    print("=" * 60)


if __name__ == "__main__":
    main()
