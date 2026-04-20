import schedule
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from ..models import HourlyData, PredictionFeedback, ModelPerformance
from .model_trainer import ModelTrainer
from .predictor import LoadPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ContinuousLearner:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.predictor = LoadPredictor()
        self.error_threshold = 20  # 20% error threshold
        
    def collect_feedback(self):
        """Collect actual loads and compare with predictions"""
        # Get predictions from last 24 hours that don't have actual load yet
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        predictions = HourlyData.objects.filter(
            date__range=[start_time, end_time],
            predicted_load__isnull=False,
            actual_load__isnull=False
        )
        
        errors = []
        for pred in predictions:
            error_percentage = abs(pred.actual_load - pred.predicted_load) / pred.actual_load * 100
            errors.append(error_percentage)
            
            # Save feedback
            PredictionFeedback.objects.create(
                predicted_load=pred.predicted_load,
                actual_load=pred.actual_load,
                error_percentage=error_percentage,
                hour_data=pred
            )
            
            # If error is high, trigger retraining
            if error_percentage > self.error_threshold:
                print(f"⚠️ High error detected: {error_percentage:.2f}% at {pred.date}")
                self.trigger_retraining()
        
        if errors:
            avg_error = np.mean(errors)
            print(f"📊 Average error last 24h: {avg_error:.2f}%")
        
        return errors
    
    def trigger_retraining(self):
        """Retrain model with new data"""
        print("🔄 Triggering model retraining...")
        
        # Retrain model in background thread
        thread = threading.Thread(target=self.retrain_model)
        thread.start()
    
    def retrain_model(self):
        """Retrain model with all available data"""
        try:
            # Get all data from database
            all_data = HourlyData.objects.filter(actual_load__isnull=False).order_by('date')
            
            if len(all_data) > 1000:  # Only retrain if enough new data
                # Export to CSV and retrain
                import pandas as pd
                data_list = []
                for record in all_data:
                    data_list.append({
                        'date': record.date,
                        'temp_1': record.temperature,
                        'rainfall_1': record.rainfall,
                        'wind_speed_1': record.wind_speed,
                        'load_1': record.actual_load,
                        # Add other fields as needed
                    })
                
                df = pd.DataFrame(data_list)
                df.to_csv('data/updated_data.csv', index=False)
                
                # Retrain model
                self.trainer.train_ensemble_model('data/updated_data.csv')
                print("✅ Model retrained successfully!")
                
                # Reload predictor
                self.predictor = LoadPredictor()
            else:
                print("Not enough new data for retraining")
                
        except Exception as e:
            print(f"Error during retraining: {e}")
    
    def daily_retraining(self):
        """Schedule daily retraining at midnight"""
        print("🔄 Starting daily model retraining...")
        self.retrain_model()
    
    def evaluate_model_performance(self):
        """Evaluate model performance weekly"""
        last_week = datetime.now() - timedelta(days=7)
        feedbacks = PredictionFeedback.objects.filter(created_at__gte=last_week)
        
        if feedbacks.exists():
            actuals = [f.actual_load for f in feedbacks]
            predictions = [f.predicted_load for f in feedbacks]
            
            mae = mean_absolute_error(actuals, predictions)
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            ModelPerformance.objects.create(
                mae=mae,
                mse=mse,
                r2_score=r2,
                samples_used=len(feedbacks)
            )
            
            print(f"📈 Weekly Performance - MAE: {mae:.2f}, R2: {r2:.4f}")

def start_continuous_learning():
    """Start the continuous learning scheduler"""
    learner = ContinuousLearner()
    
    # Schedule tasks
    schedule.every(1).hour.do(learner.collect_feedback)  # Check every hour
    schedule.every().day.at("00:00").do(learner.daily_retraining)  # Daily retraining at midnight
    schedule.every().monday.at("01:00").do(learner.evaluate_model_performance)  # Weekly evaluation
    
    print("✅ Continuous learning scheduler started")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_continuous_learning()