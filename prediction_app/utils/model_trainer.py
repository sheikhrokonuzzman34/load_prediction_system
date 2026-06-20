"""Model training utilities for the electricity load prediction project."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from .data_preprocessor import DataPreprocessor


class ModelTrainer:
    def __init__(self):
        self.app_dir = Path(__file__).resolve().parents[1]
        self.project_root = self.app_dir.parent
        self.model_dir = self.app_dir / "ml_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = DataPreprocessor(base_dir=self.app_dir)

    def _save_metrics(self, model_name: str, y_test: np.ndarray, y_pred: np.ndarray, samples: int) -> Dict[str, Any]:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {
            "model": model_name,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
            "rmse": round(rmse, 4),
            "mape": round(float(mean_absolute_percentage_error(y_test, y_pred)) * 100, 4),
            "r2_score": round(float(r2_score(y_test, y_pred)), 6),
            "samples_used": int(samples),
            "test_samples": int(len(y_test)),
        }
        with open(self.model_dir / "model_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def train_xgboost_model(self, csv_path: str | Path = "data/load_data.csv"):
        """Train the production-ready XGBoost model.

        A chronological split is used instead of random split because load data is
        a time series. This avoids future data leakage into the training set.
        """
        print("Loading and preparing data for XGBoost...")
        self.preprocessor.load_and_prepare_data(csv_path)
        X, y, feature_cols = self.preprocessor.prepare_features_for_xgboost()

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training samples: {X_train.shape}; Test samples: {X_test.shape}")
        xgb_model = xgb.XGBRegressor(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = xgb_model.predict(X_test)
        metrics = self._save_metrics("XGBoost Regressor", y_test, y_pred, samples=len(X))
        print("\nXGBoost Model Performance")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        joblib.dump(xgb_model, self.model_dir / "xgboost_model.pkl")
        joblib.dump(feature_cols, self.model_dir / "xgb_features.pkl")
        return xgb_model, metrics

    def train_lstm_model(self, csv_path: str | Path = "data/load_data.csv"):
        """Train an optional LSTM model when TensorFlow is installed.

        The web app does not depend on this model at runtime. This keeps the
        project demo-friendly even on laptops where TensorFlow is not installed.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "TensorFlow is not installed. Install it with `pip install tensorflow` "
                "before training the optional LSTM model."
            ) from exc

        print("Loading and preparing data for LSTM...")
        self.preprocessor.load_and_prepare_data(csv_path)
        X, y = self.preprocessor.prepare_sequences_for_lstm(sequence_length=24)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = keras.Sequential(
            [
                layers.Input(shape=(X.shape[1], X.shape[2])),
                layers.LSTM(96, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(48),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, name="load_output"),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        callbacks = [
            EarlyStopping(patience=8, restore_best_weights=True),
            ModelCheckpoint(str(self.model_dir / "lstm_model.keras"), save_best_only=True),
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=60,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )
        y_pred = model.predict(X_test)
        metrics = self._save_metrics("LSTM", y_test, y_pred, samples=len(X))
        return model, history, metrics

    def train_all_available_models(self, csv_path: str | Path = "data/load_data.csv"):
        """Train XGBoost and try LSTM if TensorFlow is available."""
        xgb_model, metrics = self.train_xgboost_model(csv_path)
        try:
            self.train_lstm_model(csv_path)
        except RuntimeError as exc:
            print(f"Optional LSTM skipped: {exc}")
        return xgb_model, metrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_available_models()
