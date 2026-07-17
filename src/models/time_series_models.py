"""
Time Series Forecasting Models
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Models:
1. ARIMA  — AutoRegressive Integrated Moving Average
2. SARIMA — Seasonal ARIMA (captures monthly monsoon pattern)
3. Prophet — Facebook's forecasting tool (very beginner friendly)

What these models do differently from ML models:
- ML models we built use many features (temperature, humidity etc.)
- Time series models use ONLY the past values of rainfall itself
- They are better at capturing long-term seasonal patterns
- They are more interpretable and easier to explain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

CITY = 'Mumbai'

def load_city_rainfall(city=CITY):
    """Load and prepare monthly rainfall data for time series modeling."""
    df = pd.read_csv(
        "data/raw/india_weather_1990_2024.csv",
        parse_dates=['date']
    )
    
    city_df = df[df['city'] == city].sort_values('date').copy()
    
    # Aggregate to monthly totals
    # Daily data is too noisy for ARIMA — monthly is better
    monthly = city_df.resample('M', on='date').agg({
        'precipitation_mm': 'sum',
        'temp_mean_c': 'mean',
        'humidity_max_pct': 'mean'
    }).reset_index()
    
    monthly.columns = ['date', 'rainfall_mm', 'temp_mean_c', 'humidity_pct']
    
    print(f"Monthly data: {len(monthly)} months "
          f"({monthly['date'].min().date()} to {monthly['date'].max().date()})")
    
    # Split: train on 1990-2021, test on 2022-2024
    train = monthly[monthly['date'] < '2022-01-01'].copy()
    test  = monthly[monthly['date'] >= '2022-01-01'].copy()
    
    print(f"Train: {len(train)} months | Test: {len(test)} months")
    
    return monthly, train, test


def run_prophet_model(monthly, train, test):
    """
    Prophet model — developed by Facebook, very easy to use.
    Automatically detects:
    - Trend (is rainfall increasing or decreasing over decades?)
    - Yearly seasonality (monsoon season pattern)
    - Holiday effects (if specified)
    
    You just give it a dataframe with 'ds' (date) and 'y' (value) columns.
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("Installing prophet...")
        import subprocess
        subprocess.run(["pip", "install", "prophet"], check=True)
        from prophet import Prophet
    
    print("\n" + "="*55)
    print("PROPHET MODEL")
    print("="*55)
    
    # Prophet requires columns named 'ds' and 'y'
    prophet_train = train[['date', 'rainfall_mm']].copy()
    prophet_train.columns = ['ds', 'y']
    
    # Initialize and fit model
    model = Prophet(
        yearly_seasonality=True,   # capture monsoon seasonal pattern
        weekly_seasonality=False,  # not relevant for monthly data
        daily_seasonality=False,   # not relevant for monthly data
        seasonality_mode='multiplicative',
        # multiplicative is better for monsoon
        # because rainfall variability scales with the level
        interval_width=0.95        # 95% confidence intervals
    )
    
    model.fit(prophet_train)
    print("Prophet model trained!")
    
    # Forecast for the test period
    future = model.make_future_dataframe(
    periods=len(test),
    freq='M'
)
    forecast = model.predict(future)
    
    # Extract test period predictions
    test_forecast = forecast.tail(len(test))
    y_pred = test_forecast['yhat'].values
    y_true = test['rainfall_mm'].values
    
    # Clip to 0 (no negative rainfall)
    y_pred = np.clip(y_pred, 0, None)
    
    # Evaluate
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    
    print(f"Prophet Test Results:")
    print(f"  MAE:  {mae:.2f} mm/month")
    print(f"  RMSE: {rmse:.2f} mm/month")
    print(f"  R²:   {r2:.4f}")
    
    # ─── Plot Prophet decomposition ───
    fig = model.plot_components(forecast)
    fig.suptitle(f'Prophet Model Components — {CITY} Monthly Rainfall',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/prophet_components.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # ─── Plot forecast vs actual ───
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Full historical + forecast
    ax.plot(monthly['date'], monthly['rainfall_mm'],
            color='steelblue', linewidth=1.5,
            label='Actual Rainfall', zorder=3)
    ax.plot(test['date'], y_pred,
            color='red', linewidth=2, linestyle='--',
            label='Prophet Forecast', zorder=4)
    
    # Confidence interval
    ax.fill_between(
        test_forecast['ds'],
        np.clip(test_forecast['yhat_lower'], 0, None),
        test_forecast['yhat_upper'],
        alpha=0.3, color='red', label='95% Confidence Interval'
    )
    
    # Mark test region
    ax.axvline(x=pd.Timestamp('2022-01-01'),
               color='green', linestyle=':', linewidth=2,
               label='Train/Test Split')
    
    ax.set_title(f'Prophet Forecast — {CITY} Monthly Rainfall\n'
                 f'Test RMSE: {rmse:.2f} mm | R²: {r2:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Rainfall (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/prophet_forecast.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Prophet plots saved!")
    
    return {'Model': 'Prophet', 'MAE': mae, 'RMSE': rmse, 'R2': r2}


def run_sarima_model(train, test):
    """
    SARIMA — Seasonal ARIMA model.
    
    ARIMA stands for:
    A = AutoRegressive: uses past values to predict future
    I = Integrated: differencing to make data stationary
    MA = Moving Average: uses past errors to improve prediction
    
    The S in SARIMA adds Seasonal components to capture
    the repeating monsoon pattern every 12 months.
    
    Parameters (p,d,q)(P,D,Q)[s]:
    p = AR order (how many past values to use)
    d = differencing order (usually 0 or 1)
    q = MA order (how many past errors to use)
    P,D,Q = seasonal equivalents
    s = seasonal period (12 for monthly data)
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("statsmodels not found, installing...")
        import subprocess
        subprocess.run(["pip", "install", "statsmodels"], check=True)
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    print("\n" + "="*55)
    print("SARIMA MODEL")
    print("="*55)
    
    train_values = train['rainfall_mm'].values
    
    # Fit SARIMA model
    # (1,1,1)(1,1,1)[12] is a common starting point for monthly data
    model = SARIMAX(
        train_values,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    print("Fitting SARIMA model (may take 1-2 minutes)...")
    result = model.fit(disp=False)
    print("SARIMA model fitted!")
    print(f"AIC: {result.aic:.2f} | BIC: {result.bic:.2f}")
    # Lower AIC/BIC = better model fit
    
    # Forecast test period
    n_forecast = len(test)
    forecast   = result.forecast(steps=n_forecast)
    y_pred     = np.clip(forecast, 0, None)
    y_true     = test['rainfall_mm'].values
    
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    
    print(f"SARIMA Test Results:")
    print(f"  MAE:  {mae:.2f} mm/month")
    print(f"  RMSE: {rmse:.2f} mm/month")
    print(f"  R²:   {r2:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(train['date'], train['rainfall_mm'],
            color='steelblue', linewidth=1.5, label='Training Data')
    ax.plot(test['date'],  test['rainfall_mm'],
            color='steelblue', linewidth=1.5, linestyle=':',
            label='Actual (Test)')
    ax.plot(test['date'], y_pred,
            color='darkorange', linewidth=2, linestyle='--',
            label='SARIMA Forecast')
    
    ax.axvline(x=pd.Timestamp('2022-01-01'),
               color='green', linestyle=':', linewidth=2,
               label='Train/Test Split')
    
    ax.set_title(f'SARIMA(1,1,1)(1,1,1)[12] Forecast — {CITY} Monthly Rainfall\n'
                 f'Test RMSE: {rmse:.2f} mm | R²: {r2:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Rainfall (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/sarima_forecast.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("SARIMA plot saved!")
    
    return {'Model': 'SARIMA', 'MAE': mae, 'RMSE': rmse, 'R2': r2}


if __name__ == "__main__":
    print("="*60)
    print("TIME SERIES FORECASTING")
    print(f"City: {CITY}")
    print("="*60)
    
    monthly, train, test = load_city_rainfall()
    
    results = []
    
    # Run Prophet
    prophet_result = run_prophet_model(monthly, train, test)
    results.append(prophet_result)
    
    # Run SARIMA
    sarima_result = run_sarima_model(train, test)
    results.append(sarima_result)
    
    # Compare
    print("\n" + "="*55)
    print("TIME SERIES MODEL COMPARISON")
    print("="*55)
    results_df = pd.DataFrame(results).sort_values('RMSE')
    print(results_df.to_string(index=False))
    
    results_df.to_csv('reports/time_series_results.csv', index=False)
    print("\n Time series modeling complete!")