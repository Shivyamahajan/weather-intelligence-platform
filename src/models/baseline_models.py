"""
Baseline Models for Rainfall Prediction
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Description:
    Establishes baseline performance metrics before building complex models.
    If our ML models cannot beat these baselines, they are not useful.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, handling zero values carefully."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate all performance metrics for a model.
    
    Metrics explained:
    - MAE: Average error in mm. Easy to interpret. Lower is better.
    - MSE: Penalises large errors more heavily. Lower is better.
    - RMSE: Same unit as rainfall (mm). Lower is better.
    - MAPE: Percentage error. Lower is better.
    - R2: How much variance is explained. Closer to 1.0 is better.
          0.0 means the model is as good as just predicting the mean.
          Negative means the model is worse than predicting the mean.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    
    results = {
        'Model': model_name,
        'MAE':   round(mae, 4),
        'MSE':   round(mse, 4),
        'RMSE':  round(rmse, 4),
        'MAPE':  round(mape, 4),
        'R2':    round(r2, 4)
    }
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"  MAE  (Mean Absolute Error):        {mae:.4f} mm")
    print(f"  MSE  (Mean Squared Error):         {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error):    {rmse:.4f} mm")
    print(f"  MAPE (Mean Abs Percentage Error):  {mape:.4f} %")
    print(f"  R²   (Coefficient of Determination): {r2:.4f}")
    
    return results

def persistence_baseline(df, city='Mumbai'):
    """
    Persistence model: predict tomorrow = today.
    This is the simplest possible time series baseline.
    """
    city_data = df[df['city'] == city].sort_values('date').copy()
    
    # Shift rainfall by 1 day to create predictions
    y_true = city_data['precipitation_mm'].values[1:]   # actual values
    y_pred = city_data['precipitation_mm'].values[:-1]  # yesterday's values as prediction
    
    return evaluate_model(y_true, y_pred, f'Persistence Baseline ({city})')

def monthly_average_baseline(df, city='Mumbai'):
    """
    Monthly average baseline: predict using the historical
    average rainfall for that month.
    """
    city_data = df[df['city'] == city].sort_values('date').copy()
    
    # Calculate historical monthly averages
    monthly_avg = city_data.groupby('month')['precipitation_mm'].mean()
    
    # Use monthly average as prediction
    y_pred = city_data['month'].map(monthly_avg).values
    y_true = city_data['precipitation_mm'].values
    
    return evaluate_model(y_true, y_pred, f'Monthly Average Baseline ({city})')

def zero_baseline(df, city='Mumbai'):
    """
    Zero baseline: always predict 0mm.
    On most days in India it does not rain, so this is
    surprisingly hard to beat on accuracy metrics.
    """
    city_data = df[df['city'] == city].copy()
    y_true = city_data['precipitation_mm'].values
    y_pred = np.zeros(len(y_true))
    
    return evaluate_model(y_true, y_pred, f'Zero Baseline ({city})')


if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv(
        "data/processed/india_weather_features.csv",
        parse_dates=['date']
    )
    
    print("BASELINE MODEL EVALUATION")
    print("City: Mumbai | Target: Daily Rainfall Prediction")
    
    results = []
    results.append(zero_baseline(df, 'Mumbai'))
    results.append(persistence_baseline(df, 'Mumbai'))
    results.append(monthly_average_baseline(df, 'Mumbai'))
    
    results_df = pd.DataFrame(results)
    print("\n\nBASELINE COMPARISON TABLE:")
    print(results_df.to_string(index=False))
    
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/baseline_results.csv", index=False)
    print("\nBaseline results saved to reports/baseline_results.csv")