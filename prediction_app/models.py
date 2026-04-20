from django.db import models
from django.utils import timezone

class HourlyData(models.Model):
    date = models.DateTimeField(unique=True)
    hour = models.IntegerField()
    
    # Weather and load for each hour - ALLOW NULL for all
    temperature = models.FloatField(null=True, blank=True, default=25.0)
    rainfall = models.FloatField(null=True, blank=True, default=0)
    wind_speed = models.FloatField(null=True, blank=True, default=5.0)
    actual_load = models.FloatField(null=True, blank=True)  # NULL allowed
    predicted_load = models.FloatField(null=True, blank=True)
    
    # Additional features
    residential_load = models.FloatField(null=True, blank=True)
    commercial_load = models.FloatField(null=True, blank=True)
    industrial_load = models.FloatField(null=True, blank=True)
    agricultural_load = models.FloatField(null=True, blank=True)
    religious_load = models.FloatField(null=True, blank=True)
    street_light_load = models.FloatField(null=True, blank=True)
    
    # Flags
    is_weekend = models.BooleanField(default=False)
    is_special_day = models.BooleanField(default=False)
    is_holiday = models.BooleanField(default=False)
    is_ramadan = models.BooleanField(default=False)
    
    class Meta:
        indexes = [
            models.Index(fields=['date', 'hour']),
            models.Index(fields=['date']),
        ]
    
    def __str__(self):
        load_str = str(self.actual_load) if self.actual_load else "No data"
        return f"{self.date} - Hour {self.hour}: {load_str} MW"

class PredictionFeedback(models.Model):
    predicted_load = models.FloatField()
    actual_load = models.FloatField()
    error_percentage = models.FloatField()
    hour_data = models.ForeignKey(HourlyData, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class ModelPerformance(models.Model):
    training_date = models.DateTimeField(auto_now_add=True)
    mae = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    samples_used = models.IntegerField(default=0)