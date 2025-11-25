# SARIMA model selection using AIC/BIC grid search.
# Selects optimal (p,d,q)(P,D,Q,s) orders based on lowest BIC (ties broken with AIC).

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.analysis import check_stationarity, suggest_differencing


def grid_search_sarima(
    series: pd.Series,
    d: int = 0,
    D: int = 0,
    s: int = 24,
    p_range: tuple = (0, 3),
    q_range: tuple = (0, 3),
    P_range: tuple = (0, 2),
    Q_range: tuple = (0, 2),
    max_iter: int = 50,
    return_all: bool = False
) -> dict:
    """
    Perform grid search for optimal SARIMA orders using AIC/BIC.
    
    Args:
        series: Time series to model
        d: Regular differencing order
        D: Seasonal differencing order
        s: Seasonal period (24 for hourly daily seasonality)
        p_range: Range for AR order p (start, end+1)
        q_range: Range for MA order q (start, end+1)
        P_range: Range for seasonal AR order P (start, end+1)
        Q_range: Range for seasonal MA order Q (start, end+1)
        max_iter: Maximum iterations for model fitting
    
    Returns:
        Dictionary with best model parameters and metrics
    """
    print(f"  Grid search: p={p_range}, q={q_range}, P={P_range}, Q={Q_range}")
    print(f"  Differencing: d={d}, D={D}, s={s}")
    
    results = []
    total_models = (p_range[1] - p_range[0]) * (q_range[1] - q_range[0]) * \
                  (P_range[1] - P_range[0]) * (Q_range[1] - Q_range[0])
    model_count = 0
    
    for p, q, P, Q in product(
        range(*p_range),
        range(*q_range),
        range(*P_range),
        range(*Q_range)
    ):
        model_count += 1
        if model_count % 5 == 0:
            print(f"    Testing model {model_count}/{total_models}...", end='\r')
        
        try:
            # Fit SARIMA model
            model = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(
                maxiter=max_iter,
                method='lbfgs',
                disp=False
            )
            
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'P': P,
                'D': D,
                'Q': Q,
                's': s,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'log_likelihood': fitted_model.llf,
                'converged': True
            })
        except Exception as e:
            # Skip models that fail to converge
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'P': P,
                'D': D,
                'Q': Q,
                's': s,
                'AIC': np.inf,
                'BIC': np.inf,
                'log_likelihood': -np.inf,
                'converged': False,
                'error': str(e)
            })
    
    print(f"    Tested {model_count}/{total_models} models")
    
    # Convert to DataFrame for easier sorting
    results_df = pd.DataFrame(results)
    
    # Filter to only converged models
    converged = results_df[results_df['converged']].copy()
    
    if len(converged) == 0:
        print("   Warning: No models converged!")
        return None
    
    # Sort by BIC (primary), then AIC (tie-breaker)
    converged = converged.sort_values(['BIC', 'AIC'], ascending=True)
    best = converged.iloc[0].to_dict()
    
    print(f"   Best model: SARIMA({best['p']},{best['d']},{best['q']})({best['P']},{best['D']},{best['Q']}){best['s']}")
    print(f"    BIC: {best['BIC']:.2f}, AIC: {best['AIC']:.2f}")
    
    if return_all:
        return best, converged
    return best


def select_sarima_model(
    df: pd.DataFrame,
    country_code: str,
    output_dir: Path
) -> dict:
    """
    Select optimal SARIMA model for a country's load series.
    
    Args:
        df: DataFrame with timestamp and load columns
        country_code: Country code
        output_dir: Directory to save results
    
    Returns:
        Dictionary with selected model parameters
    """
    print(f"\n{'='*60}")
    print(f"SARIMA Model Selection for {country_code}")
    print(f"{'='*60}")
    
    # Set timestamp as index
    df_analysis = df.set_index('timestamp').copy()
    series = df_analysis['load']
    
    # Use d=1, D=1 for load forecasting (standard approach)
    # This handles trend and seasonal patterns effectively
    d, D = 1, 1
    print(f"\n1. Using differencing: d={d}, D={D} (standard for load forecasting)")
    
    # Perform grid search
    print("\n2. Performing AIC/BIC grid search...")
    # Using grid: p,q,P,Q in [0,1] = 16 models per country (same as previous training)
    # Estimated time: 5-10 minutes for all 3 countries
    print("   Grid: p,q,P,Q ∈ [0,1] with d=1, D=1, s=24")
    result = grid_search_sarima(
        series,
        d=d,
        D=D,
        s=24,
        p_range=(0, 2),  # AR order: 0-1 (16 models total)
        q_range=(0, 2),   # MA order: 0-1
        P_range=(0, 2),   # Seasonal AR: 0-1
        Q_range=(0, 2),   # Seasonal MA: 0-1
        max_iter=50,
        return_all=True
    )
    
    if result is None:
        print(f"  ✗ Failed to find a suitable model for {country_code}")
        return None
    
    best_model, all_results = result
    
    # Save results
    print("\n3. Saving results...")
    results_file = output_dir / country_code / "sarima_grid_search.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save all results sorted by BIC (matching your format)
    all_results[['p', 'd', 'q', 'P', 'D', 'Q', 's', 'AIC', 'BIC']].to_csv(
        results_file, index=False
    )
    print(f"   Saved all {len(all_results)} models to {results_file}")
    print(f"   Best model: SARIMA({best_model['p']},{best_model['d']},{best_model['q']})({best_model['P']},{best_model['D']},{best_model['Q']}){best_model['s']}")
    
    return best_model


def main():
    """Perform SARIMA model selection for all countries."""
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent / "outputs"
    
    countries = ['DE', 'FR', 'ES']
    selected_models = {}
    
    for country in countries:
        input_file = processed_dir / f"{country}_clean.csv"
        
        if not input_file.exists():
            print(f" File not found: {input_file}")
            continue
        
        # Load data
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Select model
        model_params = select_sarima_model(df, country, output_dir)
        if model_params:
            selected_models[country] = model_params
    
    # Print summary
    print(f"\n{'='*60}")
    print("SARIMA Model Selection Summary")
    print(f"{'='*60}")
    print(f"\n{'Country':<10} {'Order':<20} {'Seasonal':<20} {'BIC':<12} {'AIC':<12}")
    print("-" * 80)
    
    for country, params in selected_models.items():
        order = f"({params['p']},{params['d']},{params['q']})"
        seasonal = f"({params['P']},{params['D']},{params['Q']}){params['s']}"
        print(f"{country:<10} {order:<20} {seasonal:<20} {params['BIC']:<12.2f} {params['AIC']:<12.2f}")
    
    print("-" * 80)
    print(f"\nAll results saved to country-specific folders in: {output_dir}")


if __name__ == "__main__":
    main()

