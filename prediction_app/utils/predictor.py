import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from .weather_api import WeatherAPI
from ..models import HourlyData

class LoadPredictor:
    def __init__(self):
        self.lstm_model = tf.keras.models.load_model('prediction_app/ml_models/lstm_model.h5')
        self.xgb_model = joblib.load('prediction_app/ml_models/xgboost_model.pkl')
        self.scaler = joblib.load('prediction_app/ml_models/scaler.pkl')
        self.feature_cols = joblib.load('prediction_app/ml_models/feature_columns.pkl')
        self.xgb_features = joblib.load('prediction_app/ml_models/xgb_features.pkl')
        self.weather_api = WeatherAPI()
        
    def get_previous_24h_data(self, target_datetime):
        """Get previous 24 hours data from database"""
        previous_data = []
        
        for hour_offset in range(24, 0, -1):
            prev_time = target_datetime - timedelta(hours=hour_offset)
            
            try:
                record = HourlyData.objects.get(date=prev_time)
                features = [
                    record.temperature,
                    record.rainfall,
                    record.wind_speed,
                    record.actual_load,
                    np.sin(2 * np.pi * record.hour / 24),
                    np.cos(2 * np.pi * record.hour / 24),
                    float(record.is_weekend),
                    float(record.is_holiday),
                    float(record.is_ramadan)
                ]
                # Add lag features (would need to be calculated from previous records)
                previous_data.append(features)
                
            except HourlyData.DoesNotExist:
                # If data missing, use historical average for that hour
                avg_features = self.get_historical_average(prev_time.hour)
                previous_data.append(avg_features)
        
        return np.array(previous_data).reshape(1, 24, len(self.feature_cols))
    
    def get_historical_average(self, hour):
        """Get historical average for specific hour"""
        # This should be precomputed from your dataset
        # For now, returning default values
        return [25.0, 0.5, 5.0, 150.0, 
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                0.0, 0.0, 0.0]
    
    def prepare_xgboost_features(self, target_datetime, weather_data, previous_data):
        """Prepare features for XGBoost prediction"""
        # Get the last record before target hour
        last_record = HourlyData.objects.filter(date__lt=target_datetime).order_by('-date').first()
        
        features = {
            'hour': target_datetime.hour,
            'temperature': weather_data['temperature'],
            'rainfall': weather_data['rainfall'],
            'wind_speed': weather_data['wind_speed'],
            'hour_sin': np.sin(2 * np.pi * target_datetime.hour / 24),
            'hour_cos': np.cos(2 * np.pi * target_datetime.hour / 24),
            'day_of_week': target_datetime.weekday(),
            'month': target_datetime.month,
            'is_weekend': 1 if target_datetime.weekday() >= 5 else 0,
            'is_holiday': 0,  # Need holiday calendar
            'is_ramadan': self.is_ramadan(target_datetime),
            'load_lag_1': last_record.actual_load if last_record else 150,
            'load_lag_2': 150,  # Would need previous-2
            'load_lag_3': 150,  # Would need previous-3
            'load_rolling_mean_6': 150,
            'load_rolling_mean_12': 150
        }
        
        # Convert to array in correct order
        feature_array = [features[col] for col in self.xgb_features]
        return np.array(feature_array).reshape(1, -1)
    
    def is_ramadan(self, date):
        """Check if date is during Ramadan"""
        # Implement Ramadan detection logic
        # For now, return False
        return False
    
    def predict_load(self, target_datetime):
        """
        Main prediction function
        target_datetime: datetime object for target hour (e.g., 2026-04-20 19:00:00)
        """
        print(f"Predicting load for {target_datetime}")
        
        # Step 1: Get weather forecast for target hour
        weather_data = self.weather_api.get_forecast_for_hour(target_datetime)
        print(f"Weather data: {weather_data}")
        
        # Step 2: Get previous 24 hours data
        previous_sequence = self.get_previous_24h_data(target_datetime)
        
        # Step 3: LSTM prediction
        lstm_pred_scaled = self.lstm_model.predict(previous_sequence)
        lstm_pred = self.scaler.inverse_transform(
            np.hstack([np.zeros((1, len(self.feature_cols)-1)), lstm_pred_scaled])
        )[0, -1]
        
        # Step 4: XGBoost prediction
        xgb_features = self.prepare_xgboost_features(target_datetime, weather_data, previous_sequence)
        xgb_pred = self.xgb_model.predict(xgb_features)[0]
        
        # Step 5: Ensemble (weighted average)
        final_prediction = 0.6 * lstm_pred + 0.4 * xgb_pred
        
        # Ensure prediction is reasonable
        final_prediction = max(0, min(final_prediction, 500))  # Cap between 0 and 500 MW
        
        return {
            'datetime': target_datetime,
            'predicted_load': round(final_prediction, 2),
            'lstm_prediction': round(lstm_pred, 2),
            'xgb_prediction': round(xgb_pred, 2),
            'weather_used': weather_data
        }
    
    def predict_sequential(self, start_datetime, hours_ahead=24):
        """
        Predict sequentially for next hours
        Uses previous predictions for future hours
        """
        predictions = []
        current_datetime = start_datetime
        
        for _ in range(hours_ahead):
            # Predict for current hour
            pred = self.predict_load(current_datetime)
            predictions.append(pred)
            
            # Save prediction to database for use in next iteration
            HourlyData.objects.update_or_create(
                date=current_datetime,
                defaults={
                    'hour': current_datetime.hour,
                    'temperature': pred['weather_used']['temperature'],
                    'rainfall': pred['weather_used']['rainfall'],
                    'wind_speed': pred['weather_used']['wind_speed'],
                    'predicted_load': pred['predicted_load'],
                    'is_weekend': 1 if current_datetime.weekday() >= 5 else 0,
                    'is_holiday': 0,
                    'is_ramadan': self.is_ramadan(current_datetime)
                }
            )
            
            # Move to next hour
            current_datetime += timedelta(hours=1)
        
        return predictions