from django.urls import path
from . import views

app_name = 'prediction_app'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/predict/', views.predict_single_hour, name='predict_single'),
    path('api/predict/sequential/', views.predict_sequential, name='predict_sequential'),
    path('api/feedback/', views.submit_feedback, name='feedback'),
    path('api/historical/', views.get_historical_data, name='historical'),
    path('api/performance/', views.get_model_performance, name='performance'),
]