import requests
from datetime import datetime, timedelta
import os

class WeatherAPI:
    def __init__(self, api_key=None):
        # Use your actual API key
        self.api_key = api_key or 'ffc34d91a216267b5a671fbe5d32c3e29'  # Your key
        self.use_mock = False  # Set to False to use real API
        
    def get_forecast_for_hour(self, target_datetime, lat=23.8103, lon=90.4125):
        """Get weather forecast for specific hour"""
        
        url = f"http://api.openweathermap.org/data/2.5/forecast"
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
            
            if closest_forecast and min_time_diff <= 10800:
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
        
        # Return mock data if API fails
        return self.get_mock_weather(target_datetime)
    
    def get_mock_weather(self, target_datetime):
        """Generate realistic mock weather data"""
        hour = target_datetime.hour
        
        if 6 <= hour <= 17:
            base_temp = 28
        else:
            base_temp = 22
            
        return {
            'temperature': base_temp,
            'rainfall': 0,
            'wind_speed': 5,
            'humidity': 70,
            'pressure': 1013
        }