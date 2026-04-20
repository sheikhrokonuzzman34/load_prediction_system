from django.db import models
from django.utils import timezone

class HourlyData(models.Model):
    date = models.DateTimeField(unique=True)
    hour = models.IntegerField()
    
    # Weather and load for each hour (1-24)
    temperature = models.FloatField()
    rainfall = models.FloatField()
    wind_speed = models.FloatField()
    actual_load = models.FloatField()
    predicted_load = models.FloatField(null=True, blank=True)
    
    # Additional features
    residential_load = models.FloatField(null=True)
    commercial_load = models.FloatField(null=True)
    industrial_load = models.FloatField(null=True)
    agricultural_load = models.FloatField(null=True)
    religious_load = models.FloatField(null=True)
    street_light_load = models.FloatField(null=True)
    
    # Flags
    is_weekend = models.BooleanField()
    is_special_day = models.BooleanField(default=False)
    is_holiday = models.BooleanField(default=False)
    is_ramadan = models.BooleanField(default=False)
    
    class Meta:
        indexes = [
            models.Index(fields=['date', 'hour']),
            models.Index(fields=['date']),
        ]
    
    def __str__(self):
        return f"{self.date} - Hour {self.hour}: {self.actual_load} MW"

class PredictionFeedback(models.Model):
    predicted_load = models.FloatField()
    actual_load = models.FloatField()
    error_percentage = models.FloatField()
    hour_data = models.ForeignKey(HourlyData, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class ModelPerformance(models.Model):
    training_date = models.DateTimeField(auto_now_add=True)
    mae = models.FloatField()
    mse = models.FloatField()
    r2_score = models.FloatField()
    samples_used = models.IntegerField()