"""
Machine Learning Model Training Pipeline
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Description:
    Trains, evaluates, and compares multiple ML models for
    daily rainfall prediction (regression task).
    
    Models trained:
    1. Linear Regression (baseline linear model)
    2. Ridge Regression (linear with L2 regularisation)
    3. Lasso Regression (linear with L1 regularisation)
    4. Decision Tree Regressor
    5. Random Forest Regressor
    6. XGBoost Regressor
    7. LightGBM Regressor
    8. Support Vector Regression (SVR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CITY          = 'Mumbai'      # City to train on
TARGET_COL    = 'precipitation_mm'
TEST_SIZE     = 0.2           # 20% of data for testing
RANDOM_STATE  = 42
N_CV_FOLDS    = 5             # 5-fold cross validation

# Features to use for training
# These are the features we created in Week 1's feature engineering
FEATURE_COLS = [
    # Lag features (past rainfall values)
    'precipitation_mm_lag_1',
    'precipitation_mm_lag_2',
    'precipitation_mm_lag_3',
    'precipitation_mm_lag_7',
    'precipitation_mm_lag_14',
    'precipitation_mm_lag_30',
    
    # Rolling statistics
    'precipitation_mm_rolling_mean_7d',
    'precipitation_mm_rolling_mean_30d',
    'precipitation_mm_rolling_sum_7d',
    'precipitation_mm_rolling_sum_30d',
    'precipitation_mm_rolling_std_7d',
    'precipitation_mm_rolling_std_30d',
    
    # Calendar/seasonal features
    'month_sin', 'month_cos',
    'doy_sin',   'doy_cos',
    'is_monsoon', 'is_peak_monsoon',
    'is_pre_monsoon', 'is_post_monsoon',
    
    # Weather variables
    'temp_mean_c',
    'temp_max_c',
    'temp_min_c',
    'temp_range_c',
    'humidity_max_pct',
    'humidity_min_pct',
    'humidity_range_pct',
    'wind_speed_max_kmh',
    'cloud_cover_pct',
    'evapotranspiration_mm',
]


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate(y_true, y_pred, model_name, split='Test'):
    """Calculate and print all evaluation metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    
    print(f"  [{split}] MAE={mae:.3f}mm  RMSE={rmse:.3f}mm  "
          f"R²={r2:.4f}  MAPE={mape_val:.2f}%")
    
    return {
        'Model': model_name,
        'Split': split,
        'MAE':   round(mae, 4),
        'RMSE':  round(rmse, 4),
        'R2':    round(r2, 4),
        'MAPE':  round(mape_val, 4)
    }


def load_and_prepare_data():
    """Load data and prepare train/test splits."""
    print("Loading data...")
    df = pd.read_csv(
        "data/processed/india_weather_features.csv",
        parse_dates=['date']
    )
    
    # Filter to one city for now
    city_df = df[df['city'] == CITY].sort_values('date').copy()
    print(f"City: {CITY} | Records: {len(city_df):,}")
    
    # Check which feature columns actually exist
    available_features = [col for col in FEATURE_COLS if col in city_df.columns]
    missing_features   = [col for col in FEATURE_COLS if col not in city_df.columns]
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found: {missing_features[:5]}")
    
    print(f"Using {len(available_features)} features for training")
    
    # Drop rows with NaN (from lag features)
    city_df = city_df[available_features + [TARGET_COL, 'date']].dropna()
    print(f"Records after dropping NaN: {len(city_df):,}")
    
    # Features and target
    X = city_df[available_features].values
    y = city_df[TARGET_COL].values
    dates = city_df['date'].values
    
    # TIME-BASED split — important for time series
    # Do NOT use random split for time series data
    # because future data would leak into training
    split_idx = int(len(X) * (1 - TEST_SIZE))
    
    X_train = X[:split_idx]
    X_test  = X[split_idx:]
    y_train = y[:split_idx]
    y_test  = y[split_idx:]
    dates_test = dates[split_idx:]
    
    print(f"\nTraining set: {len(X_train):,} samples "
          f"({city_df['date'].iloc[0].date()} to "
          f"{city_df['date'].iloc[split_idx-1].date()})")
    print(f"Test set:     {len(X_test):,} samples "
          f"({city_df['date'].iloc[split_idx].date()} to "
          f"{city_df['date'].iloc[-1].date()})")
    
    # Scale features
    # RobustScaler is better than StandardScaler for rainfall
    # because rainfall data has many outliers (extreme events)
    scaler  = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    return (X_train, X_test, X_train_scaled, X_test_scaled,
            y_train, y_test, dates_test, available_features, scaler)


def define_models():
    """
    Define all models with their hyperparameters.
    
    Hyperparameter explanations:
    
    Random Forest:
      - n_estimators: number of trees (more = more accurate but slower)
      - max_depth: how deep each tree can grow (prevents overfitting)
      - min_samples_split: minimum samples needed to split a node
    
    XGBoost:
      - n_estimators: number of boosting rounds
      - learning_rate: how much each tree contributes (lower = more robust)
      - max_depth: depth of each tree
      - subsample: fraction of data used for each tree (prevents overfitting)
    
    LightGBM:
      - Similar to XGBoost but faster for large datasets
      - num_leaves: controls model complexity
    """
    models = {
        'Linear Regression': LinearRegression(),
        
        'Ridge Regression': Ridge(alpha=1.0),
        # alpha controls how much to penalise large coefficients
        
        'Lasso Regression': Lasso(alpha=0.1),
        # alpha controls sparsity — forces some coefficients to exactly zero
        
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        ),
        
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,              # use all CPU cores
            random_state=RANDOM_STATE
        ),
        
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0
        ),
        
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1
        ),
        
        'SVR': SVR(
            kernel='rbf',
            C=10.0,
            epsilon=0.1,
            gamma='scale'
        ),
        # SVR works well for smaller datasets
        # For large datasets it can be slow
        
        'KNN Regressor': KNeighborsRegressor(
            n_neighbors=10,
            weights='distance',
            n_jobs=-1
        ),
    }
    return models


def train_and_evaluate_all(X_train, X_test, X_train_scaled, X_test_scaled,
                            y_train, y_test, dates_test, feature_names, scaler):
    """Train all models and collect results."""
    models    = define_models()
    all_results = []
    trained_models = {}
    predictions    = {}
    
    # Models that need scaled features (distance/gradient based)
    needs_scaling = {
        'Linear Regression', 'Ridge Regression',
        'Lasso Regression', 'SVR', 'KNN Regressor'
    }
    # Tree-based models do not need scaling
    no_scaling = {
        'Decision Tree', 'Random Forest', 'Extra Trees',
        'XGBoost', 'LightGBM'
    }
    
    os.makedirs("models", exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n{'='*55}")
        print(f"Training: {model_name}")
        print(f"{'='*55}")
        
        # Select appropriate feature version
        if model_name in needs_scaling:
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train
            X_te = X_test
        
        # Train the model
        model.fit(X_tr, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_tr)
        y_test_pred  = model.predict(X_te)
        
        # Clip negative predictions
        # Rainfall cannot be negative
        y_train_pred = np.clip(y_train_pred, 0, None)
        y_test_pred  = np.clip(y_test_pred,  0, None)
        
        # Evaluate
        train_result = evaluate(y_train, y_train_pred, model_name, 'Train')
        test_result  = evaluate(y_test,  y_test_pred,  model_name, 'Test')
        
        all_results.append(train_result)
        all_results.append(test_result)
        
        # Cross validation score
        print(f"  Running {N_CV_FOLDS}-fold cross validation...")
        cv = KFold(n_splits=N_CV_FOLDS, shuffle=False)
        cv_scores = cross_val_score(
            model, X_tr, y_train,
            cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        cv_rmse = -cv_scores.mean()
        print(f"  CV RMSE: {cv_rmse:.4f} ± {cv_scores.std():.4f}")
        
        # Save model
        model_path = f"models/{model_name.replace(' ','_').lower()}.pkl"
        joblib.dump(model, model_path)
        
        trained_models[model_name] = model
        predictions[model_name]    = y_test_pred
    
    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    
    return all_results, trained_models, predictions


def plot_model_comparison(all_results):
    """Create a comparison bar chart of all models."""
    results_df = pd.DataFrame(all_results)
    test_results = results_df[results_df['Split'] == 'Test'].copy()
    test_results = test_results.sort_values('RMSE')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(test_results)))
    
    # RMSE comparison
    bars = axes[0].barh(test_results['Model'], test_results['RMSE'],
                         color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('RMSE (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('RMSE (mm)')
    for bar, val in zip(bars, test_results['RMSE']):
        axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9)
    
    # MAE comparison
    test_results_mae = test_results.sort_values('MAE')
    colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(test_results_mae)))
    bars2 = axes[1].barh(test_results_mae['Model'], test_results_mae['MAE'],
                          color=colors2, edgecolor='white', linewidth=0.5)
    axes[1].set_title('MAE (Lower is Better)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('MAE (mm)')
    for bar, val in zip(bars2, test_results_mae['MAE']):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9)
    
    # R2 comparison
    test_results_r2 = test_results.sort_values('R2', ascending=False)
    colors3 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(test_results_r2)))
    bars3 = axes[2].barh(test_results_r2['Model'], test_results_r2['R2'],
                          color=colors3, edgecolor='white', linewidth=0.5)
    axes[2].set_title('R² Score (Higher is Better)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('R² Score')
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    for bar, val in zip(bars3, test_results_r2['R2']):
        axes[2].text(bar.get_width() + 0.002,
                     bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=9)
    
    plt.suptitle(f'ML Model Comparison — {CITY} Rainfall Prediction\n'
                 f'(Test Set Performance)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig('reports/figures/model_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nModel comparison chart saved!")


def plot_predictions_vs_actual(predictions, y_test, dates_test):
    """Plot predicted vs actual rainfall for the best models."""
    best_models = ['Random Forest', 'XGBoost', 'LightGBM']
    available   = [m for m in best_models if m in predictions]
    
    fig, axes = plt.subplots(len(available), 1,
                              figsize=(16, 5 * len(available)))
    if len(available) == 1:
        axes = [axes]
    
    # Show only last 365 days for clarity
    n_show = min(365, len(y_test))
    idx    = range(len(y_test) - n_show, len(y_test))
    
    for ax, model_name in zip(axes, available):
        y_pred = predictions[model_name]
        
        ax.plot(dates_test[-n_show:], y_test[-n_show:],
                label='Actual Rainfall', color='steelblue',
                linewidth=1.0, alpha=0.8)
        ax.plot(dates_test[-n_show:], y_pred[-n_show:],
                label=f'Predicted ({model_name})',
                color='red', linewidth=1.0, alpha=0.7, linestyle='--')
        
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        ax.set_title(f'{model_name} — Actual vs Predicted Rainfall\n'
                     f'R²={r2:.4f}, RMSE={rmse:.3f}mm',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Rainfall (mm)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Shade monsoon seasons
        ax.axhspan(100, ax.get_ylim()[1] if ax.get_ylim()[1] > 100 else 200,
                   alpha=0.05, color='blue', label='High rainfall zone')
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('reports/figures/predictions_vs_actual.png',
                dpi=150, bbox_inches='tight')
    print("Prediction charts saved!")


# ─── MAIN EXECUTION ───
if __name__ == "__main__":
    print("=" * 60)
    print("ML MODEL TRAINING PIPELINE")
    print(f"City: {CITY} | Target: {TARGET_COL}")
    print("=" * 60)
    
    # Load and prepare data
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, dates_test,
     feature_names, scaler) = load_and_prepare_data()
    
    # Train and evaluate all models
    all_results, trained_models, predictions = train_and_evaluate_all(
        X_train, X_test, X_train_scaled, X_test_scaled,
        y_train, y_test, dates_test, feature_names, scaler
    )
    
    # Save results
    results_df = pd.DataFrame(all_results)
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/ml_model_results.csv", index=False)
    
    # Print final comparison
    print("\n" + "=" * 60)
    print("FINAL TEST SET COMPARISON")
    print("=" * 60)
    test_df = results_df[results_df['Split'] == 'Test'].sort_values('RMSE')
    print(test_df.to_string(index=False))
    
    best_model = test_df.iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   RMSE: {test_df.iloc[0]['RMSE']} mm")
    print(f"   R²:   {test_df.iloc[0]['R2']}")
    
    # Create visualizations
    plot_model_comparison(all_results)
    plot_predictions_vs_actual(predictions, y_test, dates_test)
    
    print("\n Week 2 Day 1 complete! All models trained and saved.")