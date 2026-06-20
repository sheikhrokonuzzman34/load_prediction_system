# Load Prediction Project - Fix Report

## Fixed Problems

1. **ML model was not actually used**
   - Previous `predictor.py` loaded model files but still used fallback rule logic.
   - Now prediction uses the trained `xgboost_model.pkl`, `xgb_scaler.pkl`, and `xgb_features.pkl`.

2. **Empty model files removed**
   - `lstm_model.h5` and `scaler.pkl` were 0 byte files.
   - They were removed and replaced with a working XGBoost model package.

3. **TensorFlow runtime crash avoided**
   - Runtime prediction no longer imports TensorFlow.
   - LSTM training is optional and only runs if TensorFlow is installed.

4. **Weather API key security fixed**
   - Hardcoded OpenWeather API key removed.
   - Use `OPENWEATHER_API_KEY` from environment variables.
   - If no API key is available, the system uses deterministic mock weather for demos.

5. **Path issues fixed**
   - Model and dataset paths are now absolute and based on the project folder.
   - App works even if the command is executed from a different current directory.

6. **Historical lag features added for real prediction**
   - Prediction now uses previous load values and rolling averages from the dataset.
   - Sequential forecast feeds each predicted value into the next hour's lag features.

7. **Dashboard chart issue fixed**
   - Historical API now returns both `actual_load` and `predicted_load`.
   - Predictions are saved to the database, so the dashboard chart can show predicted values.

8. **Feedback error tracking improved**
   - User-submitted actual load now calculates prediction error when a predicted value exists.

9. **Bangladesh timezone applied**
   - `TIME_ZONE` changed to `Asia/Dhaka`.

10. **Model training command added**
   - Run: `python manage.py train_load_models --xgboost-only`
   - Optional LSTM training can be enabled after installing TensorFlow.

## Current Trained Model Result

Dataset: `data/load_data.csv`
Rows after hourly conversion and lag creation: 52,597 hourly samples
Train/Test Split: chronological 80/20 split
Model: XGBoost Regressor

- MAE: 5.7236 MW
- RMSE: 6.8530 MW
- MAPE: 5.3326%
- R² Score: 0.974121

## How to Run

```bash
cd core
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# Add OPENWEATHER_API_KEY in .env if you want live weather data
python manage.py migrate
python manage.py train_load_models --xgboost-only
python manage.py runserver
```

## Important Thesis Note

The current model is now working, but for final thesis defense you should clearly mention:

- The model uses 5 previous years of hourly load/weather features.
- OpenWeather free forecast gives data in 3-hour intervals, so closest forecast hour is selected.
- The current version uses XGBoost as production model; LSTM is documented as optional/future comparative model unless trained with TensorFlow.
- Final thesis should compare at least 3 models: XGBoost, Random Forest, and LSTM/GRU.
