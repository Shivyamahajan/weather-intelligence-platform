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

# Week 02 — July 14 to 19, 2026

## Day 1 (July 14) — Baseline Models & Machine Learning Pipeline

- Implemented three baseline models: Zero Baseline, Persistence Baseline, and Monthly Average Baseline.
- Developed a complete ML training pipeline for rainfall prediction.
- Trained and evaluated Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, SVR, and KNN Regressor.
- Used a time-based train-test split to prevent data leakage.
- Applied RobustScaler where required and performed 5-fold cross-validation.
- Saved trained models using Joblib.
- Generated prediction comparison plots and model performance visualizations.
- Best performing model: **XGBoost**
  - RMSE: **8.0609 mm**
  - R²: **0.7477**

---

## Day 2 (July 15) — SHAP Explainability Analysis

- Performed SHAP explainability analysis on the Random Forest model.
- Generated SHAP Feature Importance Bar Plot and SHAP Summary Plot.
- Exported feature importance rankings to CSV.
- Identified the most influential rainfall prediction features.

### Key Findings
- Evapotranspiration was the most influential feature.
- Previous day's rainfall (precipitation_mm_lag_1) was one of the strongest predictors.
- Rolling rainfall statistics improved prediction performance.
- Wind speed, humidity, cloud cover and seasonal features also contributed significantly.
- SHAP improved model interpretability by explaining feature contributions.

---

## Day 3 (July 16) — Time Series Forecasting

- Built Prophet forecasting model using monthly rainfall data.
- Built SARIMA (1,1,1)(1,1,1)[12] seasonal forecasting model.
- Compared both statistical forecasting approaches.

### Results

| Model | RMSE | R² |
|------|------|------|
| Prophet | **134.08 mm/month** | **0.8227** |
| SARIMA | 135.82 mm/month | 0.8180 |

- Prophet slightly outperformed SARIMA on the monthly rainfall forecasting task.
- Generated forecast plots and Prophet component analysis.

---

## Day 4 (July 17) — Multi-City XGBoost Analysis

- Trained XGBoost separately for every city.
- Compared prediction performance across multiple Indian cities.
- Generated multi-city performance visualization.

### Key Findings

- Jaipur achieved the lowest prediction error.
- Mumbai was the most challenging city because of highly variable monsoon rainfall.
- Model performance varied according to regional climate characteristics.

---

## Day 5 (July 18) — Model Results Notebook

- Created a comprehensive Jupyter Notebook summarizing all Week 2 experiments.
- Included:
  - Baseline comparison
  - ML model comparison
  - SHAP explainability
  - Time series forecasting
  - Multi-city analysis
  - Final observations and learnings

---

## Day 6 (July 19) — GitHub & Documentation

- Updated project documentation.
- Updated progress log with Week 2 activities.
- Organized reports, figures and trained model files.
- Prepared repository for GitHub submission.

---

# Key Technical Learnings

1. Tree-based ensemble models (XGBoost, Random Forest and LightGBM) significantly outperform linear regression for rainfall prediction.

2. Time-based train-test splitting is essential for weather forecasting to avoid data leakage.

3. SHAP provides interpretable explanations showing how each feature influences model predictions.

4. Lag features and rolling rainfall statistics are among the most informative predictors for rainfall forecasting.

5. Atmospheric variables such as evapotranspiration, cloud cover, humidity and wind speed substantially improve prediction accuracy.

6. Prophet and SARIMA effectively capture seasonal rainfall trends, while machine learning models provide higher accuracy by leveraging multiple weather variables.

---

# Next Week (Week 03)

- Build LSTM-based rainfall prediction model.
- Develop Bi-LSTM and GRU architectures.
- Compare deep learning models with classical ML models.
- Begin satellite image analysis for weather prediction.
- Continue improving the Weather Intelligence Platform.