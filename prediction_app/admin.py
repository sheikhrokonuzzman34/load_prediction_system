from django.contrib import admin
from prediction_app.models import HourlyData, PredictionFeedback, ModelPerformance
# Register your models here.

admin.site.register(HourlyData)
admin.site.register(PredictionFeedback)
admin.site.register(ModelPerformance)
