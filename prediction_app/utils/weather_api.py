"""Weather API integration for load forecasting.

OpenWeather's free 5 day / 3 hour endpoint is used when an API key is available.
If the API fails or a forecast is outside the available horizon, deterministic
mock weather is returned so the project can still run during demos/defense.
"""

from __future__ import annotations

from datetime import datetime
import math
import os
from typing import Any, Dict

import requests


class WeatherAPI:
    def __init__(self, api_key: str | None = None, use_mock: bool | None = None):
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY", "").strip()
        self.use_mock = bool(use_mock) if use_mock is not None else not bool(self.api_key)
        self.base_url = "https://api.openweathermap.org/data/2.5/forecast"

    def get_forecast_for_hour(self, target_datetime: datetime, lat: float = 23.8103, lon: float = 90.4125) -> Dict[str, Any]:
        """Return weather data for the closest available forecast hour.

        OpenWeather forecast data is available every 3 hours, so the closest
        forecast point within +/- 3 hours is used. For thesis clarity, the
        response also contains a 'source' value.
        """
        if self.use_mock:
            return self.get_mock_weather(target_datetime, source="mock_no_api_key")

        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            forecasts = data.get("list", [])

            closest_forecast = None
            min_time_diff = float("inf")
            target_naive = target_datetime.replace(tzinfo=None)
            for forecast in forecasts:
                forecast_time = datetime.fromtimestamp(forecast["dt"])
                time_diff = abs((forecast_time - target_naive).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_forecast = forecast

            if closest_forecast and min_time_diff <= 10800:
                return {
                    "temperature": float(closest_forecast.get("main", {}).get("temp", 25.0)),
                    "rainfall": float(closest_forecast.get("rain", {}).get("3h", 0.0)),
                    "wind_speed": float(closest_forecast.get("wind", {}).get("speed", 5.0)),
                    "humidity": float(closest_forecast.get("main", {}).get("humidity", 70.0)),
                    "pressure": float(closest_forecast.get("main", {}).get("pressure", 1013.0)),
                    "source": "openweather_forecast",
                }
        except requests.exceptions.RequestException as exc:
            print(f"Weather API error: {exc}")
        except (KeyError, ValueError, TypeError) as exc:
            print(f"Weather API response parsing error: {exc}")

        return self.get_mock_weather(target_datetime, source="mock_api_fallback")

    def get_mock_weather(self, target_datetime: datetime, source: str = "mock") -> Dict[str, Any]:
        """Generate realistic Bangladesh-style weather values for demo fallback."""
        hour = target_datetime.hour
        month = target_datetime.month

        # Basic seasonal approximation for Bangladesh.
        if month in [3, 4, 5]:
            base_temp = 31
            rainfall = 1.5
        elif month in [6, 7, 8, 9]:
            base_temp = 29
            rainfall = 4.0
        elif month in [11, 12, 1, 2]:
            base_temp = 22
            rainfall = 0.3
        else:
            base_temp = 27
            rainfall = 1.0

        # Diurnal temperature movement.
        temperature = base_temp + 3 * math.sin(2 * math.pi * (hour - 8) / 24)
        wind_speed = 4.0 + 1.5 * math.sin(2 * math.pi * hour / 24)

        return {
            "temperature": round(float(temperature), 2),
            "rainfall": round(float(max(rainfall, 0)), 2),
            "wind_speed": round(float(max(wind_speed, 0.5)), 2),
            "humidity": 72.0,
            "pressure": 1010.0,
            "source": source,
        }
