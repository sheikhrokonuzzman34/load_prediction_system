from __future__ import annotations

from datetime import datetime
import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import HourlyData, ModelPerformance, PredictionFeedback
from .utils.predictor import LoadPredictor

logger = logging.getLogger(__name__)
_predictor = None


def get_predictor(force_reload: bool = False):
    global _predictor
    if _predictor is None or force_reload:
        try:
            _predictor = LoadPredictor()
        except Exception as exc:
            logger.exception("Failed to initialize predictor: %s", exc)
            _predictor = None
    return _predictor


def parse_client_datetime(value: str) -> datetime:
    """Parse frontend datetime string safely."""
    if not value:
        raise ValueError("datetime is required")
    value = value.replace("T", " ").strip()
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
        try:
            dt = datetime.strptime(value, fmt)
            return timezone.make_aware(dt) if timezone.is_naive(dt) else dt
        except ValueError:
            continue
    raise ValueError("Invalid datetime format. Use YYYY-MM-DD HH:MM:SS")


def dashboard(request):
    return render(request, "dashboard.html")


@csrf_exempt
@require_http_methods(["POST"])
def predict_single_hour(request):
    """Predict and save one hourly prediction."""
    try:
        data = json.loads(request.body or "{}")
        target_datetime = parse_client_datetime(data.get("datetime"))
        lat = float(data.get("lat", 23.8103))
        lon = float(data.get("lon", 90.4125))

        predictor = get_predictor()
        if predictor is None:
            return JsonResponse({"success": False, "error": "Prediction model not available"}, status=503)

        prediction_result = predictor.predict_load(target_datetime, lat=lat, lon=lon)
        weather = prediction_result.get("weather_used", {})

        HourlyData.objects.update_or_create(
            date=target_datetime,
            defaults={
                "hour": target_datetime.hour + 1,
                "temperature": weather.get("temperature"),
                "rainfall": weather.get("rainfall"),
                "wind_speed": weather.get("wind_speed"),
                "predicted_load": prediction_result.get("predicted_load"),
                "is_weekend": target_datetime.weekday() in [4, 5],
            },
        )

        return JsonResponse({"success": True, "prediction": prediction_result})
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return JsonResponse({"success": False, "error": str(exc)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def predict_sequential(request):
    """Predict and save the next N hourly predictions."""
    try:
        data = json.loads(request.body or "{}")
        start_datetime = parse_client_datetime(data.get("start_datetime"))
        hours = int(data.get("hours", 24))
        lat = float(data.get("lat", 23.8103))
        lon = float(data.get("lon", 90.4125))

        predictor = get_predictor()
        if predictor is None:
            return JsonResponse({"success": False, "error": "Prediction model not available"}, status=503)

        predictions = predictor.predict_sequential(start_datetime, hours, lat=lat, lon=lon)
        for item in predictions:
            dt = parse_client_datetime(item["datetime"])
            weather = item.get("weather_used", {})
            HourlyData.objects.update_or_create(
                date=dt,
                defaults={
                    "hour": dt.hour + 1,
                    "temperature": weather.get("temperature"),
                    "rainfall": weather.get("rainfall"),
                    "wind_speed": weather.get("wind_speed"),
                    "predicted_load": item.get("predicted_load"),
                    "is_weekend": dt.weekday() in [4, 5],
                },
            )

        return JsonResponse({"success": True, "predictions": predictions})
    except Exception as exc:
        logger.exception("Sequential prediction error: %s", exc)
        return JsonResponse({"success": False, "error": str(exc)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """Save actual load and calculate prediction error for continuous evaluation."""
    try:
        data = json.loads(request.body or "{}")
        target_datetime = parse_client_datetime(data.get("datetime"))
        actual_load = float(data.get("actual_load"))

        record, _ = HourlyData.objects.update_or_create(
            date=target_datetime,
            defaults={
                "hour": target_datetime.hour + 1,
                "actual_load": actual_load,
                "is_weekend": target_datetime.weekday() in [4, 5],
            },
        )

        error_percentage = None
        if record.predicted_load not in [None, 0] and actual_load:
            error_percentage = abs(actual_load - record.predicted_load) / actual_load * 100
            PredictionFeedback.objects.create(
                predicted_load=record.predicted_load,
                actual_load=actual_load,
                error_percentage=error_percentage,
                hour_data=record,
            )

        return JsonResponse(
            {
                "success": True,
                "message": "Feedback submitted successfully",
                "error_percentage": round(error_percentage, 2) if error_percentage is not None else None,
            }
        )
    except Exception as exc:
        logger.exception("Feedback error: %s", exc)
        return JsonResponse({"success": False, "error": str(exc)}, status=400)


@require_http_methods(["GET"])
def get_historical_data(request):
    """Get historical/predicted load data for visualization."""
    try:
        days = int(request.GET.get("days", 7))
        start_date = timezone.now() - timezone.timedelta(days=days)
        qs = (
            HourlyData.objects.filter(date__gte=start_date)
            .order_by("date")
            .values("date", "actual_load", "predicted_load")
        )
        data_list = [
            {
                "date": item["date"].isoformat(),
                "actual_load": item["actual_load"],
                "predicted_load": item["predicted_load"],
            }
            for item in qs
        ]
        return JsonResponse({"success": True, "data": data_list})
    except Exception as exc:
        logger.exception("Historical data error: %s", exc)
        return JsonResponse({"success": True, "data": []})


@require_http_methods(["GET"])
def get_model_performance(request):
    """Get model performance metrics from DB, or from the latest training JSON."""
    try:
        performance = ModelPerformance.objects.order_by("-training_date")[:10]
        metrics = [
            {
                "date": p.training_date.strftime("%Y-%m-%d"),
                "mae": p.mae or 0,
                "mse": p.mse or 0,
                "r2": p.r2_score or 0,
                "samples": p.samples_used,
            }
            for p in performance
        ]

        if not metrics:
            predictor = get_predictor()
            if predictor and predictor.metrics:
                m = predictor.metrics
                metrics = [
                    {
                        "date": m.get("trained_at", "")[:10],
                        "mae": m.get("mae", 0),
                        "mse": m.get("mse", 0),
                        "rmse": m.get("rmse", 0),
                        "mape": m.get("mape", 0),
                        "r2": m.get("r2_score", 0),
                        "samples": m.get("samples_used", 0),
                    }
                ]
        return JsonResponse({"success": True, "performance": metrics})
    except Exception as exc:
        logger.exception("Performance error: %s", exc)
        return JsonResponse({"success": True, "performance": []})


@require_http_methods(["GET"])
def get_model_status(request):
    predictor = get_predictor()
    return JsonResponse(
        {
            "success": True,
            "model_loaded": bool(predictor and predictor.models_loaded),
            "model_used": "XGBoost Regressor" if predictor and predictor.models_loaded else "Fallback Method",
            "metrics": predictor.metrics if predictor else {},
        }
    )
