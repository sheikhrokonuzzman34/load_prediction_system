"""Prediction service used by Django views.

Fixes made from the original version:
- Removed hard dependency on TensorFlow during runtime.
- Actually loads and uses the trained XGBoost model instead of always falling
  back to a hand-written rule.
- Uses absolute paths so the app works regardless of the terminal's cwd.
- Creates lag/rolling features from historical data and updates them during
  sequential forecasts.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .data_preprocessor import DataPreprocessor
from .weather_api import WeatherAPI


class LoadPredictor:
    def __init__(self, data_csv: str | Path | None = None):
        self.app_dir = Path(__file__).resolve().parents[1]
        self.project_root = self.app_dir.parent
        self.model_dir = self.app_dir / "ml_models"
        self.data_csv = Path(data_csv) if data_csv else self.project_root / "data" / "load_data.csv"
        self.weather_api = WeatherAPI()

        self.xgb_model = None
        self.xgb_scaler = None
        self.xgb_features: List[str] | None = None
        self.models_loaded = False
        self.historical_hourly: pd.DataFrame | None = None
        self.recent_loads: Deque[float] = deque(maxlen=48)
        self.metrics: Dict[str, Any] = {}

        self.load_models()
        self.load_historical_context()

    def _file_is_valid(self, path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0

    def load_models(self) -> None:
        """Load trained XGBoost model artifacts if available."""
        xgb_path = self.model_dir / "xgboost_model.pkl"
        scaler_path = self.model_dir / "xgb_scaler.pkl"
        features_path = self.model_dir / "xgb_features.pkl"
        metrics_path = self.model_dir / "model_metrics.json"

        if all(self._file_is_valid(p) for p in [xgb_path, scaler_path, features_path]):
            try:
                self.xgb_model = joblib.load(xgb_path)
                self.xgb_scaler = joblib.load(scaler_path)
                self.xgb_features = joblib.load(features_path)
                if self._file_is_valid(metrics_path):
                    import json

                    with open(metrics_path, "r", encoding="utf-8") as f:
                        self.metrics = json.load(f)
                self.models_loaded = True
                print("✅ XGBoost load prediction model loaded successfully")
                return
            except Exception as exc:
                print(f"⚠️ Error loading XGBoost model artifacts: {exc}")

        print("⚠️ Trained XGBoost model not found. Using fallback prediction method.")
        self.models_loaded = False

    def load_historical_context(self) -> None:
        """Load historical data to calculate lag and rolling values for prediction."""
        try:
            preprocessor = DataPreprocessor(base_dir=self.app_dir)
            self.historical_hourly = preprocessor.load_and_prepare_data(self.data_csv)
            recent = self.historical_hourly["actual_load"].tail(48).tolist()
            self.recent_loads = deque([float(v) for v in recent], maxlen=48)
        except Exception as exc:
            print(f"⚠️ Could not load historical data for lag features: {exc}")
            self.historical_hourly = None
            self.recent_loads = deque([120.0] * 12, maxlen=48)

    def _normalise_datetime(self, dt: datetime) -> datetime:
        return dt.replace(tzinfo=None, minute=0, second=0, microsecond=0)

    def _hour_number(self, dt: datetime) -> int:
        # Training data uses 1-24. midnight => 1, 23:00 => 24.
        return dt.hour + 1

    def _calendar_flags(self, dt: datetime) -> Dict[str, int]:
        # This is a safe default. For final thesis, replace or extend it with a
        # Bangladesh government holiday + Ramadan calendar table.
        return {
            "is_weekend": 1 if dt.weekday() in [4, 5] else 0,  # Friday/Saturday in Bangladesh context
            "is_holiday": 0,
            "is_ramadan": 0,
        }

    def _get_lag_values(self, target_datetime: datetime, recent_loads: Iterable[float] | None = None) -> Tuple[float, float, float, float, float]:
        """Return lag_1, lag_2, lag_3, rolling_mean_6 and rolling_mean_12."""
        dt = self._normalise_datetime(target_datetime)

        # If the target exists inside the historical dataset, use true previous values.
        if self.historical_hourly is not None and not self.historical_hourly.empty:
            hist = self.historical_hourly
            previous = hist[hist["date"] < pd.Timestamp(dt)].tail(12)
            if len(previous) >= 3:
                values = previous["actual_load"].astype(float).tolist()
                lag_1 = values[-1]
                lag_2 = values[-2]
                lag_3 = values[-3]
                roll_6 = float(np.mean(values[-6:])) if len(values) >= 6 else float(np.mean(values))
                roll_12 = float(np.mean(values[-12:])) if len(values) >= 12 else float(np.mean(values))
                return lag_1, lag_2, lag_3, roll_6, roll_12

        loads = list(recent_loads if recent_loads is not None else self.recent_loads)
        if len(loads) < 3:
            loads = [120.0, 120.0, 120.0]
        lag_1 = float(loads[-1])
        lag_2 = float(loads[-2])
        lag_3 = float(loads[-3])
        roll_6 = float(np.mean(loads[-6:]))
        roll_12 = float(np.mean(loads[-12:]))
        return lag_1, lag_2, lag_3, roll_6, roll_12

    def _build_xgb_features(self, target_datetime: datetime, weather_data: Dict[str, Any], recent_loads: Iterable[float] | None = None) -> pd.DataFrame:
        dt = self._normalise_datetime(target_datetime)
        hour = self._hour_number(dt)
        flags = self._calendar_flags(dt)
        lag_1, lag_2, lag_3, roll_6, roll_12 = self._get_lag_values(dt, recent_loads=recent_loads)

        row = {
            "hour": hour,
            "temperature": float(weather_data.get("temperature", 25.0)),
            "rainfall": float(weather_data.get("rainfall", 0.0)),
            "wind_speed": float(weather_data.get("wind_speed", 5.0)),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "is_weekend": flags["is_weekend"],
            "is_holiday": flags["is_holiday"],
            "is_ramadan": flags["is_ramadan"],
            "load_lag_1": lag_1,
            "load_lag_2": lag_2,
            "load_lag_3": lag_3,
            "load_rolling_mean_6": roll_6,
            "load_rolling_mean_12": roll_12,
        }

        columns = self.xgb_features or DataPreprocessor.xgboost_feature_columns()
        return pd.DataFrame([[row[col] for col in columns]], columns=columns)

    def fallback_prediction(self, target_datetime: datetime, weather_data: Dict[str, Any]) -> float:
        """Rule-based fallback prediction when the trained model is not available."""
        hour = target_datetime.hour
        is_weekend = target_datetime.weekday() in [4, 5]

        if 6 <= hour <= 9:
            base_load = 155
        elif 18 <= hour <= 21:
            base_load = 190
        elif 23 <= hour or hour <= 5:
            base_load = 75
        else:
            base_load = 125

        temp_adjustment = (float(weather_data.get("temperature", 25)) - 25) * 2.2
        rain_adjustment = float(weather_data.get("rainfall", 0)) * 3.0
        wind_adjustment = abs(float(weather_data.get("wind_speed", 5)) - 5) * 1.5
        weekend_adjustment = -18 if is_weekend else 0
        final_load = base_load + temp_adjustment + rain_adjustment + wind_adjustment + weekend_adjustment
        return max(40.0, min(float(final_load), 500.0))

    def predict_load(self, target_datetime: datetime, lat: float = 23.8103, lon: float = 90.4125, recent_loads: Iterable[float] | None = None) -> Dict[str, Any]:
        """Predict one hourly load value."""
        target_datetime = self._normalise_datetime(target_datetime)
        weather_data = self.weather_api.get_forecast_for_hour(target_datetime, lat=lat, lon=lon)

        model_used = "Fallback Method"
        if self.models_loaded and self.xgb_model is not None and self.xgb_scaler is not None:
            try:
                features = self._build_xgb_features(target_datetime, weather_data, recent_loads=recent_loads)
                features_scaled = self.xgb_scaler.transform(features.values)
                final_prediction = float(self.xgb_model.predict(features_scaled)[0])
                model_used = "XGBoost Regressor"
            except Exception as exc:
                print(f"⚠️ XGBoost prediction failed: {exc}; using fallback")
                final_prediction = self.fallback_prediction(target_datetime, weather_data)
        else:
            final_prediction = self.fallback_prediction(target_datetime, weather_data)

        final_prediction = max(40.0, min(float(final_prediction), 500.0))
        return {
            "datetime": target_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_load": round(final_prediction, 2),
            "weather_used": weather_data,
            "model_used": model_used,
            "model_metrics": self.metrics,
        }

    def predict_sequential(self, start_datetime: datetime, hours_ahead: int = 24, lat: float = 23.8103, lon: float = 90.4125) -> List[Dict[str, Any]]:
        """Predict a sequence and feed each predicted value into future lag features."""
        predictions: List[Dict[str, Any]] = []
        current_datetime = self._normalise_datetime(start_datetime)
        recent = deque(self.recent_loads, maxlen=48)

        for _ in range(max(1, min(int(hours_ahead), 168))):
            pred = self.predict_load(current_datetime, lat=lat, lon=lon, recent_loads=recent)
            predictions.append(pred)
            recent.append(float(pred["predicted_load"]))
            current_datetime += timedelta(hours=1)

        return predictions
