import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from datetime import datetime, timedelta
from .weather_api import WeatherAPI

class LoadPredictor:
    def __init__(self):
        self.weather_api = WeatherAPI()
        self.models_loaded = False
        
        # Try to load models, if not exists, use fallback method
        self.load_models()
    
    def load_models(self):
        """Load trained models if they exist"""
        model_dir = 'prediction_app/ml_models/'
        
        # Check if model files exist
        lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
        
        if os.path.exists(lstm_path) and os.path.exists(xgb_path):
            try:
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                self.xgb_model = joblib.load(xgb_path)
                self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
                self.models_loaded = True
                print("✅ Models loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading models: {e}")
                self.models_loaded = False
        else:
            print("⚠️ Model files not found. Using fallback prediction method.")
            self.models_loaded = False
    
    def fallback_prediction(self, target_datetime, weather_data):
        """Simple fallback prediction when ML models are not available"""
        hour = target_datetime.hour
        is_weekend = target_datetime.weekday() >= 5
        
        # Time of day pattern
        if 6 <= hour <= 9:  # Morning peak
            base_load = 180
        elif 18 <= hour <= 21:  # Evening peak
            base_load = 200
        elif 23 <= hour or hour <= 5:  # Night off-peak
            base_load = 80
        else:
            base_load = 130
        
        # Weather adjustments
        temp_adjustment = (weather_data['temperature'] - 25) * 2
        rain_adjustment = weather_data['rainfall'] * 5
        wind_adjustment = abs(weather_data['wind_speed'] - 5) * 3
        
        # Weekend adjustment
        weekend_adjustment = -30 if is_weekend else 0
        
        # Time of day peak adjustment
        if 18 <= hour <= 21:
            time_adjustment = 40
        elif 6 <= hour <= 9:
            time_adjustment = 30
        else:
            time_adjustment = 0
        
        final_load = base_load + temp_adjustment + rain_adjustment + wind_adjustment + weekend_adjustment + time_adjustment
        
        # Ensure reasonable bounds
        final_load = max(50, min(final_load, 400))
        
        return final_load
    
    def predict_load(self, target_datetime):
        """
        Main prediction function - returns prediction without saving to DB
        """
        print(f"Predicting load for {target_datetime}")
        
        # Get weather forecast for target hour
        weather_data = self.weather_api.get_forecast_for_hour(target_datetime)
        print(f"Weather data: {weather_data}")
        
        # Make prediction based on available model
        if self.models_loaded:
            try:
                # ML prediction logic here
                final_prediction = self.fallback_prediction(target_datetime, weather_data)  # Placeholder
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}, using fallback")
                final_prediction = self.fallback_prediction(target_datetime, weather_data)
        else:
            # Use fallback method
            final_prediction = self.fallback_prediction(target_datetime, weather_data)
        
        # Ensure prediction is reasonable
        final_prediction = max(50, min(final_prediction, 500))
        
        return {
            'datetime': target_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_load': round(final_prediction, 2),
            'weather_used': weather_data,
            'model_used': 'Fallback Method (ML models not trained yet)'
        }
    
    def predict_sequential(self, start_datetime, hours_ahead=24):
        """
        Predict sequentially for next hours without saving to database
        """
        predictions = []
        current_datetime = start_datetime
        
        for i in range(hours_ahead):
            # Predict for current hour
            pred = self.predict_load(current_datetime)
            predictions.append(pred)
            
            # Move to next hour
            current_datetime += timedelta(hours=1)
        
        return predictions