"""Continuous evaluation helpers.

This module keeps feedback/error tracking stable. Full online retraining from DB is
left as future work because the training CSV is a daily wide-format dataset while
feedback is stored as hourly rows. The current safe implementation evaluates errors
and can retrain from the canonical CSV dataset when requested.
"""

from __future__ import annotations

from datetime import timedelta
import threading

import numpy as np
from django.utils import timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..models import HourlyData, ModelPerformance, PredictionFeedback
from .model_trainer import ModelTrainer
from .predictor import LoadPredictor


class ContinuousLearner:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.predictor = LoadPredictor()
        self.error_threshold = 20.0

    def collect_feedback(self):
        """Compare saved predictions with actual values from the last 24 hours."""
        end_time = timezone.now()
        start_time = end_time - timedelta(hours=24)
        records = HourlyData.objects.filter(
            date__range=[start_time, end_time],
            predicted_load__isnull=False,
            actual_load__isnull=False,
        )

        errors = []
        for record in records:
            if not record.actual_load:
                continue
            error_percentage = abs(record.actual_load - record.predicted_load) / record.actual_load * 100
            errors.append(error_percentage)
            PredictionFeedback.objects.get_or_create(
                hour_data=record,
                predicted_load=record.predicted_load,
                actual_load=record.actual_load,
                defaults={"error_percentage": error_percentage},
            )

        return errors

    def trigger_retraining(self, csv_path="data/load_data.csv"):
        """Retrain in a background thread from the canonical training CSV."""
        thread = threading.Thread(target=self.retrain_model, kwargs={"csv_path": csv_path}, daemon=True)
        thread.start()

    def retrain_model(self, csv_path="data/load_data.csv"):
        self.trainer.train_xgboost_model(csv_path)
        self.predictor = LoadPredictor()
        return True

    def evaluate_model_performance(self):
        """Evaluate weekly performance from user-submitted feedback."""
        last_week = timezone.now() - timedelta(days=7)
        feedbacks = PredictionFeedback.objects.filter(created_at__gte=last_week)
        if not feedbacks.exists():
            return None

        actuals = np.array([f.actual_load for f in feedbacks], dtype=float)
        predictions = np.array([f.predicted_load for f in feedbacks], dtype=float)
        mae = float(mean_absolute_error(actuals, predictions))
        mse = float(mean_squared_error(actuals, predictions))
        r2 = float(r2_score(actuals, predictions)) if len(actuals) > 1 else 0.0

        return ModelPerformance.objects.create(
            mae=mae,
            mse=mse,
            r2_score=r2,
            samples_used=len(feedbacks),
        )
