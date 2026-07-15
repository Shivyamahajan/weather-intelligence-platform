"""
SHAP Explainability Analysis
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Description:
    Uses SHAP (SHapley Additive exPlanations) to explain why
    our Random Forest model makes the predictions it makes.
    
    What is SHAP?
    SHAP values tell us how much each feature contributed to
    pushing a prediction higher or lower than the average prediction.
    
    Example: If the model predicts 25mm rainfall for tomorrow,
    SHAP might tell us:
    - cloud_cover_pct contributed +15mm (high cloud cover → more rain)
    - humidity_max_pct contributed +8mm (high humidity → more rain)  
    - month_sin contributed +5mm (July = monsoon season)
    - pressure_hpa contributed -3mm (high pressure → less rain)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
CITY         = 'Mumbai'
FEATURE_COLS = [
    'precipitation_mm_lag_1', 'precipitation_mm_lag_2',
    'precipitation_mm_lag_3', 'precipitation_mm_lag_7',
    'precipitation_mm_lag_14','precipitation_mm_lag_30',
    'precipitation_mm_rolling_mean_7d',
    'precipitation_mm_rolling_mean_30d',
    'precipitation_mm_rolling_sum_7d',
    'precipitation_mm_rolling_sum_30d',
    'precipitation_mm_rolling_std_7d',
    'precipitation_mm_rolling_std_30d',
    'month_sin', 'month_cos',
    'doy_sin',   'doy_cos',
    'is_monsoon', 'is_peak_monsoon',
    'is_pre_monsoon', 'is_post_monsoon',
    'temp_mean_c', 'temp_max_c', 'temp_min_c', 'temp_range_c',
    'humidity_max_pct', 'humidity_min_pct', 'humidity_range_pct',
    'wind_speed_max_kmh', 'pressure_hpa',
    'cloud_cover_pct', 'evapotranspiration_mm',
]

def run_shap_analysis():
    # Load data
    df = pd.read_csv(
        "data/processed/india_weather_features.csv",
        parse_dates=['date']
    )
    
    city_df = df[df['city'] == CITY].sort_values('date')
    available = [c for c in FEATURE_COLS if c in city_df.columns]
    
    X = city_df[available].dropna()
    y = city_df.loc[X.index, 'precipitation_mm']
    
    # Use a sample for SHAP (full dataset can be slow)
    sample_size = min(2000, len(X))
    X_sample    = X.sample(n=sample_size, random_state=42)
    
    # Load the trained Random Forest model
    model = joblib.load("models/random_forest.pkl")
    
    print("Calculating SHAP values (this may take 2-3 minutes)...")
    
    # TreeExplainer is the fastest SHAP method for tree-based models
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print("SHAP values calculated!")
    
    # ─── Plot 1: SHAP Summary Bar Plot ───
    # Shows the average absolute SHAP value for each feature
    # = which features are most important overall
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type='bar',
        feature_names=available,
        show=False,
        max_display=20
    )
    plt.title(f'SHAP Feature Importance — {CITY} Rainfall Prediction\n'
              f'(Average |SHAP Value| = Average Impact on Prediction)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_importance_bar.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP bar chart saved!")
    
    # ─── Plot 2: SHAP Dot Plot ───
    # More detailed: shows direction of impact too
    # Red dots = high feature value, blue = low feature value
    # Right of center = pushes prediction higher
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=available,
        show=False,
        max_display=20
    )
    plt.title(f'SHAP Summary Plot — {CITY} Rainfall Prediction\n'
              f'Red = High Value, Blue = Low Value | '
              f'Right = Increases Prediction',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary_dot.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP dot plot saved!")
    
    # ─── Top Features Summary ───
    feature_importance = pd.DataFrame({
        'feature': available,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*50)
    print(feature_importance.head(10).to_string(index=False))
    
    feature_importance.to_csv(
        'reports/shap_feature_importance.csv', index=False
    )
    print("\nSHAP analysis complete!")
    
    return feature_importance


if __name__ == "__main__":
    run_shap_analysis()