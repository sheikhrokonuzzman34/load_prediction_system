from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from .models import HourlyData, PredictionFeedback, ModelPerformance
from .utils.predictor import LoadPredictor
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

# Initialize predictor
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = LoadPredictor()
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            _predictor = None
    return _predictor

def make_aware(dt):
    """Make datetime timezone aware"""
    if dt.tzinfo is None:
        return timezone.make_aware(dt)
    return dt

def dashboard(request):
    """Main dashboard view"""
    return render(request, 'dashboard.html')

@csrf_exempt
@require_http_methods(["POST"])
def predict_single_hour(request):
    """Predict load for a single hour"""
    try:
        data = json.loads(request.body)
        target_datetime = datetime.strptime(data['datetime'], '%Y-%m-%d %H:%M:%S')
        target_datetime = make_aware(target_datetime)
        
        # Get predictor
        predictor = get_predictor()
        if not predictor:
            return JsonResponse({
                'success': False,
                'error': 'Prediction model not available'
            }, status=503)
        
        # Make prediction (without saving to database)
        prediction_result = predictor.predict_load(target_datetime)
        
        return JsonResponse({
            'success': True,
            'prediction': prediction_result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def predict_sequential(request):
    """Predict for next 24 hours sequentially"""
    try:
        data = json.loads(request.body)
        start_datetime = datetime.strptime(data['start_datetime'], '%Y-%m-%d %H:%M:%S')
        start_datetime = make_aware(start_datetime)
        hours = data.get('hours', 24)
        
        predictor = get_predictor()
        if not predictor:
            return JsonResponse({
                'success': False,
                'error': 'Prediction model not available'
            }, status=503)
        
        predictions = predictor.predict_sequential(start_datetime, hours)
        
        return JsonResponse({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Sequential prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """Submit actual load for prediction"""
    try:
        data = json.loads(request.body)
        target_datetime = datetime.strptime(data['datetime'], '%Y-%m-%d %H:%M:%S')
        target_datetime = make_aware(target_datetime)
        actual_load = float(data['actual_load'])
        
        # Update or create record with actual load
        record, created = HourlyData.objects.update_or_create(
            date=target_datetime,
            defaults={
                'hour': target_datetime.hour,
                'actual_load': actual_load,
                'is_weekend': 1 if target_datetime.weekday() >= 5 else 0
            }
        )
        
        return JsonResponse({
            'success': True,
            'message': 'Feedback submitted successfully'
        })
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@require_http_methods(["GET"])
def get_historical_data(request):
    """Get historical load data for visualization"""
    try:
        days = int(request.GET.get('days', 7))
        start_date = timezone.now() - timezone.timedelta(days=days)
        
        data = HourlyData.objects.filter(
            date__gte=start_date,
            actual_load__isnull=False  # Only get records with actual load
        ).order_by('date').values('date', 'actual_load')
        
        data_list = []
        for item in data:
            data_list.append({
                'date': item['date'].isoformat(),
                'actual_load': item['actual_load']
            })
        
        return JsonResponse({
            'success': True,
            'data': data_list
        })
        
    except Exception as e:
        logger.error(f"Historical data error: {str(e)}")
        return JsonResponse({
            'success': True,
            'data': []
        })

@require_http_methods(["GET"])
def get_model_performance(request):
    """Get model performance metrics"""
    try:
        performance = ModelPerformance.objects.order_by('-training_date')[:10]
        
        metrics = [{
            'date': p.training_date.strftime('%Y-%m-%d'),
            'mae': p.mae if p.mae else 0,
            'mse': p.mse if p.mse else 0,
            'r2': p.r2_score if p.r2_score else 0,
            'samples': p.samples_used
        } for p in performance]
        
        return JsonResponse({
            'success': True,
            'performance': metrics
        })
        
    except Exception as e:
        logger.error(f"Performance error: {str(e)}")
        return JsonResponse({
            'success': True,
            'performance': []  # Return empty array
        })