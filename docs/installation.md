# Installation Guide

## Prerequisites
- Python 3.11+
- Git
- Ollama (for GenAI features)
- 8GB RAM minimum (16GB recommended)
- 15GB free disk space

## Step 1: Clone Repository
```bash
git clone https://github.com/YOUR-USERNAME/weather-intelligence-platform.git
cd weather-intelligence-platform
```

## Step 2: Python Environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Step 3: Collect Data
```bash
python src/data_collection/collect_openmeteo.py
python src/preprocessing/feature_engineering.py
```

## Step 4: Train Models
```bash
python src/models/train_ml_models.py
python src/models/lstm_model.py
python src/evaluation/final_evaluation.py
```

## Step 5: Setup GenAI (Part 02)
```bash
# Install Ollama from ollama.com
ollama pull Qwen2.5:1.5b
python src/genai/build_knowledge_base.py
```

## Step 6: Run Application
```bash
# Terminal 1 - FastAPI
uvicorn src.app.main:app --reload --port 8000

# Terminal 2 - Streamlit
streamlit run src/app/final_app.py

# Or with Docker
docker-compose up --build
```