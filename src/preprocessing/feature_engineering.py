"""
Feature Engineering Script
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026
Description: Creates features for ML model training from raw weather data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load and validate raw weather data."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"Loaded {len(df):,} records from {filepath}")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate strategies.
    - Rainfall: fill with 0 (no rain is valid)
    - Temperature: forward fill then backward fill
    - Other: interpolate
    """
    df = df.copy()
    
    print("\nHandling missing values...")
    print(f"Before: {df.isnull().sum().sum()} missing values")
    
    # Rainfall NaN means no rain
    df['precipitation_mm'] = df['precipitation_mm'].fillna(0)
    
    # Temperature: interpolate within city groups
    temp_cols = ['temp_max_c', 'temp_min_c', 'temp_mean_c']
    for col in temp_cols:
        df[col] = df.groupby('city')[col].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )
    
    # Other variables: interpolate
    other_cols = ['humidity_max_pct', 'humidity_min_pct', 'wind_speed_max_kmh',
                   'cloud_cover_pct']
    for col in other_cols:
        df[col] = df.groupby('city')[col].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )
    
    print(f"After: {df.isnull().sum().sum()} missing values")
    return df

def create_lag_features(df: pd.DataFrame, 
                         target_col: str = 'precipitation_mm',
                         lags: list = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Create lag features — previous values of a variable.
    Example: rainfall yesterday (lag 1), last week (lag 7) etc.
    These are critical for time series prediction.
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby('city')[target_col].shift(lag)
    
    print(f"Created {len(lags)} lag features for {target_col}")
    return df

def create_rolling_features(df: pd.DataFrame,
                             target_col: str = 'precipitation_mm',
                             windows: list = [7, 14, 30, 90]) -> pd.DataFrame:
    """
    Create rolling window statistics — average/sum over past N days.
    Example: total rainfall in last 7 days, last 30 days etc.
    """
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}d'] = df.groupby('city')[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        # Rolling sum
        df[f'{target_col}_rolling_sum_{window}d'] = df.groupby('city')[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
        # Rolling standard deviation (variability)
        df[f'{target_col}_rolling_std_{window}d'] = df.groupby('city')[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    print(f"Created rolling features for windows: {windows}")
    return df

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features that capture seasonal patterns.
    Monsoon follows a strict calendar pattern so these are very powerful.
    """
    df = df.copy()
    
    # Sine and cosine encoding of month (captures cyclical nature of seasons)
    # This is better than using month as a number because Dec and Jan are close
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day of year encoding
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Binary season flags
    df['is_pre_monsoon'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['is_post_monsoon'] = df['month'].isin([10, 11]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    # Peak monsoon months
    df['is_peak_monsoon'] = df['month'].isin([7, 8]).astype(int)
    
    print("Created calendar and seasonal features")
    return df

def create_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived temperature features."""
    df = df.copy()
    
    # Temperature range (max - min)
    df['temp_range_c'] = df['temp_max_c'] - df['temp_min_c']
    
    # Humidity range
    df['humidity_range_pct'] = df['humidity_max_pct'] - df['humidity_min_pct']
    
    # Heat index approximation
    df['heat_index'] = df['temp_mean_c'] + 0.33 * (df['humidity_max_pct'] / 100 * 6.105 * 
                        np.exp(17.27 * df['temp_mean_c'] / (237.7 + df['temp_mean_c']))) - 4.0
    
    print("Created temperature derived features")
    return df

def encode_city(df: pd.DataFrame) -> pd.DataFrame:
    """Encode city as numerical features."""
    df = df.copy()
    city_dummies = pd.get_dummies(df['city'], prefix='city', dtype=int)
    df = pd.concat([df, city_dummies], axis=1)
    print(f"City encoding: created {len(city_dummies.columns)} city dummy variables")
    return df

def prepare_ml_dataset(df: pd.DataFrame, 
                         target_city: str = None,
                         drop_na: bool = True) -> pd.DataFrame:
    """
    Final dataset preparation for ML models.
    Optionally filter to one city.
    """
    df = df.copy()
    
    if target_city:
        df = df[df['city'] == target_city].copy()
        print(f"Filtered to city: {target_city}")
    
    # Drop rows with NaN (created by lag features at start of series)
    if drop_na:
        before = len(df)
        df = df.dropna()
        print(f"Dropped {before - len(df)} rows with NaN values (from lag features)")
    
    print(f"Final dataset shape: {df.shape}")
    return df


# ─── Run the full pipeline ───
if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Load data
    df = load_raw_data("data/raw/india_weather_1990_2024.csv")
    
    # Apply all transformations
    df = handle_missing_values(df)
    df = create_lag_features(df, target_col='precipitation_mm', lags=[1,2,3,7,14,30])
    df = create_rolling_features(df, target_col='precipitation_mm', windows=[7,14,30,90])
    df = create_calendar_features(df)
    df = create_temperature_features(df)
    df = encode_city(df)
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/india_weather_features.csv", index=False)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nFeature list ({len(df.columns)} total):")
    for col in sorted(df.columns):
        print(f"  - {col}")