# System Architecture

## Part 01: Weather Prediction Engine

Data Sources (Open-Meteo, ERA5, IMD)
↓
Data Collection & Preprocessing
↓
Feature Engineering (60+ features)
↓
Model Training & Evaluation
├── ML Models (XGBoost, RF, LightGBM)
├── Time Series (SARIMA, Prophet)
├── Deep Learning (LSTM, BiLSTM, GRU)
└── Computer Vision (CNN, MobileNetV2)
↓
FastAPI REST Backend
↓
Streamlit Web Dashboard


## Part 02: GenAI Intelligence Layer

Weather Documents (IMD, NASA, NDMA)
↓
Document Processing & Chunking
↓
Embedding (all-MiniLM-L6-v2)
↓
ChromaDB Vector Store
↓
RAG Pipeline (LangChain + MMR)
↓
┌─────────────────────────────┐
│ Multi-Agent Workflow │
│ ┌─────────────────────┐ │
│ │ Supervisor Agent │ │
│ └──────────┬──────────┘ │
│ ┌─────────┼────────┐ │
│ ↓ ↓ ↓ │
│ Research Prediction Analysis│
│ Agent Agent Agent │
│ └─────────┬────────┘ │
│ Response Agent │
│ └─────────────────────┘ │
└─────────────────────────────┘
↓
MCP Tool Interface
↓
Streamlit GenAI Dashboard