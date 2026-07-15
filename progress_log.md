# Progress Log

## Week 01 — July 7 to 11, 2026

### Day 1 (July 7) — Environment Setup
- Installed Python 3.11, VS Code, Git
- Created GitHub repository: weather-intelligence-platform
- Set up project folder structure
- Created virtual environment
- Installed all required libraries via requirements.txt

### Day 2 (July 8) — Data Collection
- Studied Open-Meteo API documentation
- Understood what data is needed for monsoon analysis
- Collected 35 years (1990–2024) of daily weather data for 6 Indian cities
- Dataset size: 76704 records, 20+ variables per record
- Saved to data/raw/ folder

### Day 3 (July 9) — Exploratory Data Analysis
- Built complete EDA notebook (notebooks/01_EDA.ipynb)
- Created 6+ visualizations: annual rainfall trends, monthly monsoon patterns,
  temperature trends, correlation heatmap, extreme events analysis, India map
- Key finding: Mumbai receives 5-6x more monsoon rainfall than Jaipur
- Key finding: Extreme rainfall events are showing an increasing trend since 2000

### Day 4 (July 10) — Feature Engineering
- Built feature engineering pipeline (src/preprocessing/feature_engineering.py)
- Created lag features (1, 2, 3, 7, 14, 30 days)
- Created rolling window features (7, 14, 30, 90 days)
- Created cyclical calendar features (sine/cosine encoding of month)
- Created temperature derived features (range, heat index)
- Final processed dataset: 76,704 rows × 58 features

### Day 5 (July 11) — Documentation & GitHub
- Written detailed README.md
- Pushed all code to GitHub
- Updated progress log

### What I Learned This Week
1. Open-Meteo API is extremely easy to use and gives high-quality historical data
2. Indian cities show dramatically different monsoon patterns — Mumbai gets 2500mm/year, Jaipur gets 300mm
3. Feature engineering (lag features, rolling averages) is critical for time series prediction
4. Cyclical encoding of months (sin/cos) is better than treating month as a plain number
5. Extreme rainfall events across India appear to be increasing — a signal of climate change

### Next Week Plan (July 14–18)
- Build baseline ML models (Linear Regression, Random Forest, XGBoost)
- Implement time-series models (ARIMA, SARIMA, Prophet)
- Start model evaluation and comparison

# Week 02 – Day 1 (14 July 2026)

## Objective
Build and compare machine learning models for daily rainfall prediction using the processed weather dataset.

## Work Completed
- Implemented baseline models:
  - Zero Baseline
  - Persistence Baseline
  - Monthly Average Baseline
- Evaluated baseline models using MAE, RMSE, MAPE, and R².
- Developed a complete machine learning training pipeline.
- Trained and evaluated the following regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Extra Trees Regressor
  - XGBoost Regressor
  - LightGBM Regressor
  - Support Vector Regressor (SVR)
  - KNN Regressor
- Used a time-based train-test split to prevent data leakage.
- Applied RobustScaler for models requiring feature scaling.
- Performed 5-fold cross-validation for model evaluation.
- Saved all trained models using Joblib.
- Generated model comparison visualizations and prediction plots.

## Key Results
### Baseline Models
| Model | RMSE | R² |
|-------|------|------|
| Zero Baseline | 15.1567 | -0.1452 |
| Persistence Baseline | 11.9018 | 0.2939 |
| Monthly Average Baseline | 12.0860 | 0.2718 |

### Best Machine Learning Models
| Model | RMSE | R² |
|-------|------|------|
| XGBoost | **8.0609** | **0.7477** |
| LightGBM | 8.0782 | 0.7466 |
| Random Forest | 8.1114 | 0.7445 |

## Observations
- Tree-based ensemble models significantly outperformed linear regression models.
- XGBoost achieved the best prediction accuracy.
- The trained ML models substantially improved upon all baseline models.
- Model artifacts and evaluation reports were successfully generated for future use.

## Files Created
- src/models/baseline_models.py
- src/models/train_ml_models.py
- reports/baseline_results.csv
- reports/ml_model_results.csv
- reports/figures/model_comparison.png
- reports/figures/predictions_vs_actual.png
- models/*.pkl

## Week 02 – Day 2 (July 15)

- Performed SHAP explainability analysis on the Random Forest rainfall prediction model.
- Generated SHAP feature importance bar chart and SHAP summary plot.
- Exported feature importance rankings to `reports/shap_feature_importance.csv`.

### Key Findings

- Evapotranspiration was the most influential feature for rainfall prediction.
- Previous day's rainfall (precipitation_mm_lag_1) was one of the strongest predictors, showing temporal dependence.
- Rolling rainfall statistics over the previous week improved predictive performance.
- Wind speed, humidity, cloud cover, and seasonal features also contributed to rainfall prediction.
- SHAP analysis improved model interpretability by explaining how individual features influenced predictions.