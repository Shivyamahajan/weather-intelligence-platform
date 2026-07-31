"""
Prediction Tracker
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Description:
    Saves every prediction made by the API to a CSV file.
    This allows you to:
    1. Monitor model performance over time
    2. Identify if predictions are systematically wrong
    3. Build a dataset of real-world predictions to evaluate later
    4. Include usage statistics in your research paper
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any


class PredictionTracker:
    """Tracks all predictions made by the weather platform."""
    
    def __init__(self, log_file: str = "reports/prediction_log.csv"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Define columns
        self.columns = [
            'timestamp', 'city', 'prediction_date',
            'predicted_mm', 'rainfall_category',
            'temp_max_c', 'humidity_pct',
            'cloud_cover_pct', 'prev_rain_mm',
            'model_used', 'response_time_ms'
        ]
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(log_file):
            pd.DataFrame(columns=self.columns).to_csv(
                log_file, index=False
            )
            print(f"Prediction log created: {log_file}")
    
    def log(self, prediction_data: Dict[str, Any],
            response_time_ms: float):
        """Log a single prediction to CSV."""
        row = {
            'timestamp':         datetime.now().isoformat(),
            'city':              prediction_data.get('city', ''),
            'prediction_date':   prediction_data.get('date', ''),
            'predicted_mm':      prediction_data.get(
                'predicted_rainfall_mm', 0
            ),
            'rainfall_category': prediction_data.get(
                'rainfall_category', ''
            ),
            'temp_max_c':        prediction_data.get('temp_max_c', 0),
            'humidity_pct':      prediction_data.get(
                'humidity_max_pct', 0
            ),
            'cloud_cover_pct':   prediction_data.get(
                'cloud_cover_pct', 0
            ),
            'prev_rain_mm':      prediction_data.get(
                'prev_day_rainfall_mm', 0
            ),
            'model_used':        prediction_data.get('model_used', ''),
            'response_time_ms':  round(response_time_ms, 2)
        }
        
        new_row = pd.DataFrame([row])
        new_row.to_csv(
            self.log_file, mode='a', header=False, index=False
        )
    
    def get_statistics(self) -> Dict:
        """Return summary statistics about predictions made."""
        if not os.path.exists(self.log_file):
            return {}
        
        df = pd.read_csv(self.log_file)
        
        if len(df) == 0:
            return {'total_predictions': 0}
        
        return {
            'total_predictions':    len(df),
            'unique_cities':        df['city'].nunique(),
            'avg_predicted_mm':     round(df['predicted_mm'].mean(), 2),
            'max_predicted_mm':     round(df['predicted_mm'].max(), 2),
            'avg_response_time_ms': round(
                df['response_time_ms'].mean(), 2
            ),
            'most_queried_city':    df['city'].mode()[0],
            'predictions_by_city':  df['city'].value_counts().to_dict()
        }