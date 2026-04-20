import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from .data_preprocessor import DataPreprocessor

class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        
    def train_lstm_model(self, csv_path='data/your_original_data.csv'):
        print("Loading and preparing data...")
        self.preprocessor.load_and_prepare_data(csv_path)
        X, y = self.preprocessor.prepare_sequences_for_lstm(sequence_length=24)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {X_train.shape}, Test samples: {X_test.shape}")
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, name='load_output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('prediction_app/ml_models/lstm_model.h5', save_best_only=True)
        ]
        
        # Train model
        print("Training LSTM model...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nLSTM Model Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        return model, history
    
    def train_xgboost_model(self, csv_path='data/your_original_data.csv'):
        print("Loading and preparing data for XGBoost...")
        self.preprocessor.load_and_prepare_data(csv_path)
        X, y, feature_cols = self.preprocessor.prepare_features_for_xgboost()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train XGBoost
        print("Training XGBoost model...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = xgb_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nXGBoost Model Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save model
        joblib.dump(xgb_model, 'prediction_app/ml_models/xgboost_model.pkl')
        joblib.dump(feature_cols, 'prediction_app/ml_models/xgb_features.pkl')
        
        return xgb_model
    
    def train_ensemble_model(self, csv_path='data/your_original_data.csv'):
        """Train ensemble of LSTM and XGBoost"""
        lstm_model, _ = self.train_lstm_model(csv_path)
        xgb_model = self.train_xgboost_model(csv_path)
        
        print("\n✅ Both models trained successfully!")
        return lstm_model, xgb_model

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_ensemble_model()