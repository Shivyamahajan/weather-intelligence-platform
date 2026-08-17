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

## Week 03 — July 20 to 25, 2026

### Day 1 (July 20) — LSTM / BiLSTM / GRU Deep Learning
- Understood concept of sequences vs flat feature tables
- Built LSTM model with 128→64 units, Dropout, EarlyStopping
- Built Bidirectional LSTM (reads sequences forward + backward)
- Built GRU model (simpler alternative to LSTM)
- Used Huber loss (better than MSE for skewed rainfall data)
- EarlyStopping prevented overfitting automatically

### Day 2 (July 21) — CNN Satellite Image Classification
- Downloaded/created cloud image dataset with 4 categories
- Built CNN from scratch (Conv2D → BatchNorm → MaxPool blocks)
- Built Transfer Learning model with MobileNetV2 pretrained on ImageNet
- Applied data augmentation (rotation, flip, zoom, brightness)
- Transfer learning significantly outperformed CNN from scratch

### Day 3 (July 22) — Complete Model Comparison
- Built master comparison across ALL models from Weeks 2 and 3
- Documented key findings and personal observations
- Identified best model overall and per category

### Day 4 (July 23) — FastAPI Backend
- Built REST API with FastAPI
- Endpoints: /health, /cities, /predict/rainfall, /weather/current
- Pydantic validation for all inputs
- Live weather fetching from Open-Meteo API
- Automatic Swagger docs at localhost:8000/docs

### Day 5 (July 24) — Streamlit Frontend
- Built 5-page web application in pure Python
- Dashboard: live current weather
- Prediction: interactive rainfall forecasting
- Historical Analysis: city rainfall trends
- Model Performance: comparison charts
- About: project documentation

### Day 6 (July 25) — Docker + GitHub
- Created Dockerfile and docker-compose.yml
- Pushed all Week 3 code to GitHub
- Updated documentation

### Key Technical Learnings This Week
1. LSTM sequences reshape how you think about input — 
   instead of one row, you feed a window of time
2. Transfer learning is remarkably powerful — MobileNetV2 
   achieved better accuracy on cloud images despite being 
   trained on completely different images (everyday objects)
3. FastAPI auto-generates Swagger UI — this is genuinely 
   impressive and made testing much faster
4. Streamlit lets you build real web apps in pure Python — 
   the barrier between data science and web development has
   almost disappeared with tools like this
5. EarlyStopping is essential for deep learning — 
   without it models overfit quickly

## Week 05 — August 3 to 7, 2026

### Day 1 (August 3) — Ollama Setup + First LLM Interaction

- Installed Ollama and configured Qwen2.5:1.5b for local LLM inference
- Installed LangChain, ChromaDB, and sentence-transformers libraries
- Tested the local LLM through Ollama and Python
- Built the first LangChain chain (prompt → LLM → parser)
- Observed that an LLM without RAG can hallucinate specific rainfall data

### Day 2 (August 4) — Knowledge Base Construction

- Collected 3 text documents for the knowledge base:
  IMD rainfall classification, monsoon science, and project results
- Built document loading pipeline using TextLoader and PyPDFLoader
- Implemented RecursiveCharacterTextSplitter with 500-character chunks and 50-character overlap
- Created HuggingFace embeddings using all-MiniLM-L6-v2 with 384 dimensions
- Built and persisted ChromaDB vector store
- Tested retrieval with multiple sample queries and verified relevant chunks were returned

### Day 3 (August 5) — RAG Pipeline

- Built the complete WeatherRAGPipeline class
- Implemented retrieve → format context → prompt → generate flow
- Created structured RAGResponse dataclass with timing metrics
- Tested the pipeline with multiple weather questions
- Integrated ML prediction data into RAG prompts

### Day 4 (August 6) — Conversational Assistant

- Added ConversationBufferWindowMemory with a 5-exchange window
- Built a conversational prompt that includes previous conversation history
- Implemented explain_prediction() to integrate ML predictions with the LLM
- Added conversation saving to text files
- Tested multi-turn conversations and verified that memory works correctly

### Day 5 (August 7) — Integration + GitHub

- Built the Part 02 demo notebook (04_Part02_RAG_Demo.ipynb)
- Compared RAG and No-RAG responses using multiple weather questions
- Demonstrated conversational memory across multiple turns
- Integrated Part 01 ML predictions with the GenAI assistant
- Fixed the ChromaDB path configuration for notebook execution
- Verified the RAG pipeline using the 14-chunk weather knowledge base
- Prepared Part 02 Week 1 work for GitHub

### Key Technical Learnings This Week

1. Temperature setting affects LLM response behaviour — lower temperature
   values such as 0.1 are more suitable for factual responses.

2. Chunk size affects retrieval quality — chunks that are too small may
   lose context, while very large chunks may reduce retrieval precision.

3. Embedding model choice matters — all-MiniLM-L6-v2 provides a good
   balance between speed and semantic retrieval quality for this project.

4. RAG provides access to project-specific knowledge by retrieving relevant
   document chunks before generating the answer.

5. Conversation memory makes interactions more natural by preserving
   previous turns, but increases prompt length and therefore has a
   production performance tradeoff.

6. Local LLM inference with Qwen2.5:1.5b through Ollama allows the
   application to operate without relying on a paid external LLM API.

