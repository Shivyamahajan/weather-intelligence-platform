"""
Multi-City Model Training and Comparison
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Description:
    Trains the best model (XGBoost or Random Forest based on Day 1 results)
    across all 8 cities and compares performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import joblib, os, warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'precipitation_mm_lag_1', 'precipitation_mm_lag_2',
    'precipitation_mm_lag_3', 'precipitation_mm_lag_7',
    'precipitation_mm_lag_14','precipitation_mm_lag_30',
    'precipitation_mm_rolling_mean_7d',
    'precipitation_mm_rolling_mean_30d',
    'precipitation_mm_rolling_sum_7d',
    'precipitation_mm_rolling_std_7d',
    'month_sin', 'month_cos',
    'doy_sin', 'doy_cos',
    'is_monsoon', 'is_peak_monsoon',
    'temp_mean_c', 'temp_range_c',
    'humidity_max_pct', 'wind_speed_max_kmh',
    'pressure_hpa', 'cloud_cover_pct',
]

def train_city(city, df):
    """Train and evaluate XGBoost for a single city."""
    city_df = df[df['city'] == city].sort_values('date').copy()
    
    available = [c for c in FEATURE_COLS if c in city_df.columns]
    data      = city_df[available + ['precipitation_mm', 'date']].dropna()
    
    X = data[available].values
    y = data['precipitation_mm'].values
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        n_jobs=-1, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    
    y_pred = np.clip(model.predict(X_test), 0, None)
    
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    # Annual rainfall stats
    annual_actual = data['precipitation_mm'].sum() / (len(data) / 365)
    
    return {
        'City': city,
        'Annual_Rainfall_mm': round(annual_actual, 0),
        'Test_MAE':  round(mae, 3),
        'Test_RMSE': round(rmse, 3),
        'Test_R2':   round(r2, 4),
        'Train_Samples': split,
        'Test_Samples':  len(X) - split
    }


if __name__ == "__main__":
    df = pd.read_csv(
        "data/processed/india_weather_features.csv",
        parse_dates=['date']
    )
    
    cities  = df['city'].unique()
    results = []
    
    print("Training XGBoost across all cities...\n")
    
    for city in cities:
        print(f"  Processing {city}...", end=' ')
        result = train_city(city, df)
        results.append(result)
        print(f"RMSE={result['Test_RMSE']} mm | R²={result['Test_R2']}")
    
    results_df = pd.DataFrame(results).sort_values('Test_RMSE')
    
    print("\n" + "="*65)
    print("MULTI-CITY XGBOOST PERFORMANCE SUMMARY")
    print("="*65)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('reports/multi_city_results.csv', index=False)
    
    # ─── Visualization ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sort by RMSE
    df_sorted = results_df.sort_values('Test_RMSE')
    
    axes[0].barh(
        df_sorted['City'], df_sorted['Test_RMSE'],
        color='steelblue', alpha=0.8
    )
    axes[0].set_title('RMSE by City (Lower = Better)',
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel('RMSE (mm/day)')
    
    df_sorted2 = results_df.sort_values('Test_R2', ascending=False)
    axes[1].barh(
        df_sorted2['City'], df_sorted2['Test_R2'],
        color='forestgreen', alpha=0.8
    )
    axes[1].set_title('R² Score by City (Higher = Better)',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('R² Score')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('XGBoost Performance Across Indian Cities\n'
                 'Weather Intelligence Platform',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/multi_city_performance.png',
                dpi=150, bbox_inches='tight')
    print("\nMulti-city comparison chart saved!")
    print("✅ Multi-city analysis complete!")