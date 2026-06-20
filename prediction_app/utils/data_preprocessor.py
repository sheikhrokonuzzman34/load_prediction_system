"""Data preparation utilities for station/area based electricity load forecasting.

The original dataset is stored in a wide daily format: one row per day and
columns such as temp_1, rainfall_1, wind_speed_1, load_1 ... load_24.
This module converts the data into an hourly time-series and creates lag,
rolling and calendar features that can be used by tree models and LSTM models.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
        self.model_dir = self.base_dir / "ml_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = MinMaxScaler()
        self.feature_columns: List[str] | None = None
        self.df_hourly: pd.DataFrame | None = None

    def load_and_prepare_data(self, csv_path: str | Path = "data/load_data.csv") -> pd.DataFrame:
        """Load the raw CSV and return an hourly dataframe with engineered features."""
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            # Prefer project root/data first, then app/data if someone passes a relative path.
            project_root = self.base_dir.parent
            candidate = project_root / csv_path
            csv_path = candidate if candidate.exists() else self.base_dir / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "date" not in df.columns:
            raise ValueError("CSV must contain a 'date' column")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        hourly_data = []
        for _, row in df.iterrows():
            base_date = row["date"]
            for hour in range(1, 25):
                temp_col = f"temp_{hour}"
                rainfall_col = f"rainfall_{hour}"
                wind_col = f"wind_speed_{hour}"
                load_col = f"load_{hour}"

                if not {temp_col, rainfall_col, wind_col, load_col}.issubset(row.index):
                    continue
                if pd.isna(row[load_col]):
                    continue

                timestamp = base_date + timedelta(hours=hour - 1)
                hourly_data.append(
                    {
                        "date": timestamp,
                        "hour": hour,  # 1-24 format, matching the original CSV convention
                        "temperature": float(row[temp_col]),
                        "rainfall": float(row[rainfall_col]),
                        "wind_speed": float(row[wind_col]),
                        "actual_load": float(row[load_col]),
                        "residential_load": float(row.get("Residential_load", 0) or 0),
                        "commercial_load": float(row.get("Commercial_load", 0) or 0),
                        "industrial_load": float(row.get("Industrial_load", 0) or 0),
                        "agricultural_load": float(row.get("Agricultural_load", 0) or 0),
                        "religious_load": float(row.get("Religious_educational_load", 0) or 0),
                        "street_light_load": float(row.get("Street_light_load", 0) or 0),
                        "is_weekend": int(row.get("Is_weekend", 0) or 0),
                        "is_special_day": int(row.get("Is_special_day", 0) or 0),
                        "is_holiday": int(row.get("Is_holiday", 0) or 0),
                        "is_ramadan": int(row.get("Is_ramadan", 0) or 0),
                    }
                )

        if not hourly_data:
            raise ValueError("No hourly records could be created from the CSV")

        self.df_hourly = pd.DataFrame(hourly_data).sort_values("date").reset_index(drop=True)
        self.create_time_features()
        return self.df_hourly

    def create_time_features(self) -> None:
        """Create time, lag and rolling features for hourly forecasting."""
        if self.df_hourly is None or self.df_hourly.empty:
            raise ValueError("No hourly dataframe found. Run load_and_prepare_data() first.")

        df = self.df_hourly.sort_values("date").copy()
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

        df["load_lag_1"] = df["actual_load"].shift(1)
        df["load_lag_2"] = df["actual_load"].shift(2)
        df["load_lag_3"] = df["actual_load"].shift(3)
        df["load_rolling_mean_6"] = df["actual_load"].rolling(6).mean()
        df["load_rolling_mean_12"] = df["actual_load"].rolling(12).mean()

        self.df_hourly = df.dropna().reset_index(drop=True)

    @staticmethod
    def xgboost_feature_columns() -> List[str]:
        return [
            "hour",
            "temperature",
            "rainfall",
            "wind_speed",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
            "is_ramadan",
            "load_lag_1",
            "load_lag_2",
            "load_lag_3",
            "load_rolling_mean_6",
            "load_rolling_mean_12",
        ]

    @staticmethod
    def lstm_feature_columns() -> List[str]:
        return [
            "temperature",
            "rainfall",
            "wind_speed",
            "actual_load",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "is_ramadan",
            "load_lag_1",
            "load_lag_2",
            "load_lag_3",
            "load_rolling_mean_6",
            "load_rolling_mean_12",
        ]

    def prepare_sequences_for_lstm(self, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare scaled sequential samples for LSTM training."""
        if self.df_hourly is None:
            raise ValueError("Load data first by calling load_and_prepare_data().")

        feature_cols = self.lstm_feature_columns()
        self.feature_columns = feature_cols
        data = self.df_hourly[feature_cols].values

        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        X, y = [], []
        load_idx = feature_cols.index("actual_load")
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i : i + sequence_length])
            y.append(data_scaled[i + sequence_length, load_idx])

        joblib.dump(self.scaler, self.model_dir / "lstm_scaler.pkl")
        joblib.dump(feature_cols, self.model_dir / "lstm_feature_columns.pkl")
        return np.array(X), np.array(y)

    def prepare_features_for_xgboost(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare scaled tabular features for XGBoost."""
        if self.df_hourly is None:
            raise ValueError("Load data first by calling load_and_prepare_data().")

        feature_cols = self.xgboost_feature_columns()
        X = self.df_hourly[feature_cols].values
        y = self.df_hourly["actual_load"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, self.model_dir / "xgb_scaler.pkl")
        joblib.dump(feature_cols, self.model_dir / "xgb_features.pkl")
        return X_scaled, y, feature_cols
