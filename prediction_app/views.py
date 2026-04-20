from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from .models import HourlyData, PredictionFeedback
from .utils.predictor import LoadPredictor
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)
predictor = LoadPredictor()

def dashboard(request):
    """Main dashboard view"""
    return render(request, 'prediction_app/dashboard.html')

@csrf_exempt
@require_http_methods(["POST"])
def predict_single_hour(request):
    """Predict load for a single hour"""
    try:
        data = json.loads(request.body)
        target_datetime = datetime.strptime(data['datetime'], '%Y-%m-%d %H:%M:%S')
        
        # Make prediction
        prediction_result = predictor.predict_load(target_datetime)
        
        # Save prediction to database
        HourlyData.objects.update_or_create(
            date=target_datetime,
            defaults={
                'hour': target_datetime.hour,
                'temperature': prediction_result['weather_used']['temperature'],
                'rainfall': prediction_result['weather_used']['rainfall'],
                'wind_speed': prediction_result['weather_used']['wind_speed'],
                'predicted_load': prediction_result['predicted_load'],
                'is_weekend': 1 if target_datetime.weekday() >= 5 else 0,
                'is_holiday': 0,  # Add holiday detection
                'is_ramadan': 0   # Add Ramadan detection
            }
        )
        
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
        hours = data.get('hours', 24)
        
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
        actual_load = float(data['actual_load'])
        
        # Update the record with actual load
        record, created = HourlyData.objects.update_or_create(
            date=target_datetime,
            defaults={'actual_load': actual_load}
        )
        
        # Get the predicted load if exists
        if record.predicted_load:
            error_percentage = abs(actual_load - record.predicted_load) / actual_load * 100
            
            PredictionFeedback.objects.create(
                predicted_load=record.predicted_load,
                actual_load=actual_load,
                error_percentage=error_percentage,
                hour_data=record
            )
            
            # Check if error is high and trigger learning
            if error_percentage > 20:
                # Schedule retraining in background
                from .utils.continuous_learner import ContinuousLearner
                learner = ContinuousLearner()
                learner.trigger_retraining()
        
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
            date__gte=start_date
        ).order_by('date').values('date', 'actual_load', 'predicted_load')
        
        return JsonResponse({
            'success': True,
            'data': list(data)
        })
        
    except Exception as e:
        logger.error(f"Historical data error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@require_http_methods(["GET"])
def get_model_performance(request):
    """Get model performance metrics"""
    try:
        from .models import ModelPerformance
        performance = ModelPerformance.objects.order_by('-training_date')[:10]
        
        metrics = [{
            'date': p.training_date.strftime('%Y-%m-%d'),
            'mae': p.mae,
            'mse': p.mse,
            'r2': p.r2_score,
            'samples': p.samples_used
        } for p in performance]
        
        return JsonResponse({
            'success': True,
            'performance': metrics
        })
        
    except Exception as e:
        logger.error(f"Performance error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)