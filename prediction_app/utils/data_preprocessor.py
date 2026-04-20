import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime, timedelta
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        
    def load_and_prepare_data(self, csv_path='data/your_original_data.csv'):
        """Load CSV and prepare for training"""
        df = pd.read_csv(csv_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Reshape from wide format to long format (24 hours per day)
        hourly_data = []
        
        for idx, row in df.iterrows():
            base_date = row['date']
            
            for hour in range(1, 25):
                temp_col = f'temp_{hour}'
                rainfall_col = f'rainfall_{hour}'
                wind_col = f'wind_speed_{hour}'
                load_col = f'load_{hour}'
                
                if temp_col in row and not pd.isna(row[temp_col]):
                    hourly_record = {
                        'date': base_date + timedelta(hours=hour-1),
                        'hour': hour,
                        'temperature': row[temp_col],
                        'rainfall': row[rainfall_col],
                        'wind_speed': row[wind_col],
                        'actual_load': row[load_col],
                        'residential_load': row.get('Residential_load', 0),
                        'commercial_load': row.get('Commercial_load', 0),
                        'industrial_load': row.get('Industrial_load', 0),
                        'agricultural_load': row.get('Agricultural_load', 0),
                        'religious_load': row.get('Religious_educational_load', 0),
                        'street_light_load': row.get('Street_light_load', 0),
                        'is_weekend': row.get('Is_weekend', 0),
                        'is_special_day': row.get('Is_special_day', 0),
                        'is_holiday': row.get('Is_holiday', 0),
                        'is_ramadan': row.get('Is_ramadan', 0)
                    }
                    hourly_data.append(hourly_record)
        
        self.df_hourly = pd.DataFrame(hourly_data)
        
        # Create time-based features
        self.create_time_features()
        
        return self.df_hourly
    
    def create_time_features(self):
        """Create additional time-based features"""
        self.df_hourly['hour_sin'] = np.sin(2 * np.pi * self.df_hourly['hour'] / 24)
        self.df_hourly['hour_cos'] = np.cos(2 * np.pi * self.df_hourly['hour'] / 24)
        self.df_hourly['day_of_week'] = self.df_hourly['date'].dt.dayofweek
        self.df_hourly['month'] = self.df_hourly['date'].dt.month
        self.df_hourly['day_of_year'] = self.df_hourly['date'].dt.dayofyear
        
        # Rolling averages (previous hours)
        self.df_hourly = self.df_hourly.sort_values('date')
        self.df_hourly['load_lag_1'] = self.df_hourly['actual_load'].shift(1)
        self.df_hourly['load_lag_2'] = self.df_hourly['actual_load'].shift(2)
        self.df_hourly['load_lag_3'] = self.df_hourly['actual_load'].shift(3)
        self.df_hourly['load_rolling_mean_6'] = self.df_hourly['actual_load'].rolling(6).mean()
        self.df_hourly['load_rolling_mean_12'] = self.df_hourly['actual_load'].rolling(12).mean()
        
        # Drop rows with NaN from shifting
        self.df_hourly = self.df_hourly.dropna()
    
    def prepare_sequences_for_lstm(self, sequence_length=24):
        """Prepare sequences for LSTM training"""
        feature_cols = [
            'temperature', 'rainfall', 'wind_speed', 'actual_load',
            'hour_sin', 'hour_cos', 'is_weekend', 'is_holiday', 'is_ramadan',
            'load_lag_1', 'load_lag_2', 'load_lag_3',
            'load_rolling_mean_6', 'load_rolling_mean_12'
        ]
        
        self.feature_columns = feature_cols
        data = self.df_hourly[feature_cols].values
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i+sequence_length])
            y.append(data_scaled[i+sequence_length, 3])  # actual_load is at index 3
        
        # Save scaler and feature columns
        joblib.dump(self.scaler, 'prediction_app/ml_models/scaler.pkl')
        joblib.dump(feature_cols, 'prediction_app/ml_models/feature_columns.pkl')
        
        return np.array(X), np.array(y)
    
    def prepare_features_for_xgboost(self):
        """Prepare features for XGBoost"""
        feature_cols = [
            'hour', 'temperature', 'rainfall', 'wind_speed',
            'hour_sin', 'hour_cos', 'day_of_week', 'month',
            'is_weekend', 'is_holiday', 'is_ramadan',
            'load_lag_1', 'load_lag_2', 'load_lag_3',
            'load_rolling_mean_6', 'load_rolling_mean_12'
        ]
        
        X = self.df_hourly[feature_cols].values
        y = self.df_hourly['actual_load'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, 'prediction_app/ml_models/xgb_scaler.pkl')
        
        return X_scaled, y, feature_cols