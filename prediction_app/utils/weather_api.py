import requests
from datetime import datetime, timedelta
from django.conf import settings
import os
from dotenv import load_dotenv

load_dotenv()

class WeatherAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY', 'YOUR_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_forecast_for_hour(self, target_datetime, lat=23.8103, lon=90.4125):
        """
        Get weather forecast for specific hour
        target_datetime: datetime object for target hour (e.g., 2026-04-20 19:00:00)
        """
        # OpenWeatherMap gives 3-hour intervals, we need to interpolate
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Find closest forecast
            closest_forecast = None
            min_time_diff = float('inf')
            
            for forecast in data['list']:
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                time_diff = abs((forecast_time - target_datetime).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_forecast = forecast
            
            if closest_forecast and min_time_diff <= 10800:  # Within 3 hours
                weather_data = {
                    'temperature': closest_forecast['main']['temp'],
                    'rainfall': closest_forecast.get('rain', {}).get('3h', 0),
                    'wind_speed': closest_forecast['wind']['speed'],
                    'humidity': closest_forecast['main']['humidity'],
                    'pressure': closest_forecast['main']['pressure']
                }
                return weather_data
            
        except requests.exceptions.RequestException as e:
            print(f"Weather API error: {e}")
        
        # Return default values if API fails
        return {
            'temperature': 25.0,
            'rainfall': 0,
            'wind_speed': 5.0,
            'humidity': 70,
            'pressure': 1013
        }
    
    def get_24h_forecast(self, start_datetime, lat=23.8103, lon=90.4125):
        """Get 24 hours forecast starting from start_datetime"""
        forecasts = {}
        for hour in range(24):
            target_time = start_datetime + timedelta(hours=hour)
            forecasts[target_time] = self.get_forecast_for_hour(target_time, lat, lon)
        return forecasts