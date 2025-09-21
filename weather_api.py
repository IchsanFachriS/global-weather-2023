
import flask
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

# Create the Flask app
app = Flask(__name__)

# Load the pre-trained model
@app.before_first_request
def load_model():
    global temperature_model, air_quality_model
    
    # Load temperature prediction model
    with open('models/temperature_model.pkl', 'rb') as f:
        temperature_model = pickle.load(f)
    
    # Load air quality prediction model
    with open('models/air_quality_model.pkl', 'rb') as f:
        air_quality_model = pickle.load(f)

# Define the prediction endpoint for temperature
@app.route('/api/predict/temperature', methods=['POST'])
def predict_temperature():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Extract required features
    try:
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        day_of_year = int(data['day_of_year'])
        
        # Calculate cyclical features for day of year
        sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
        cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Create feature array
        features = np.array([[latitude, longitude, sin_day, cos_day]])
        
        # Make prediction
        prediction = temperature_model.predict(features)[0]
        
        # Return prediction
        return jsonify({
            'prediction': round(float(prediction), 2),
            'unit': 'celsius'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Invalid input. Please provide latitude, longitude, and day_of_year.'
        }), 400

# Define the prediction endpoint for air quality
@app.route('/api/predict/air-quality', methods=['POST'])
def predict_air_quality():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Extract required features
    try:
        # Required features
        temperature = float(data['temperature_celsius'])
        humidity = float(data['humidity'])
        wind_kph = float(data['wind_kph'])
        pressure_mb = float(data.get('pressure_mb', 1013.25))  # Default value if not provided
        cloud = int(data.get('cloud', 0))  # Default value if not provided
        country = data.get('country', 'Unknown')
        climate_zone = data.get('climate_zone', 'Temperate')
        
        # Create a dataframe with the input data
        input_df = pd.DataFrame({
            'temperature_celsius': [temperature],
            'humidity': [humidity],
            'wind_kph': [wind_kph],
            'pressure_mb': [pressure_mb],
            'cloud': [cloud],
            'country': [country],
            'climate_zone': [climate_zone]
        })
        
        # Make prediction
        prediction = air_quality_model.predict(input_df)[0]
        
        # Map prediction to air quality category
        categories = {
            1: 'Good',
            2: 'Moderate',
            3: 'Unhealthy for Sensitive Groups',
            4: 'Unhealthy',
            5: 'Very Unhealthy',
            6: 'Hazardous'
        }
        
        category = categories.get(prediction, 'Unknown')
        
        # Return prediction
        return jsonify({
            'prediction': int(prediction),
            'category': category
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Invalid input. Please check your input data.'
        }), 400

# Root endpoint for API information
@app.route('/')
def api_info():
    return jsonify({
        'name': 'Weather and Air Quality Prediction API',
        'version': '1.0',
        'endpoints': [
            {
                'path': '/api/predict/temperature',
                'method': 'POST',
                'description': 'Predicts temperature based on latitude, longitude, and day of year',
                'parameters': ['latitude', 'longitude', 'day_of_year']
            },
            {
                'path': '/api/predict/air-quality',
                'method': 'POST',
                'description': 'Predicts air quality index based on weather parameters',
                'parameters': ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'cloud', 'country', 'climate_zone']
            }
        ]
    })

# Run the Flask app
if __name__ == '__main__':
    # Ensure the model directory exists
    os.makedirs('models', exist_ok=True)
    
    # For demonstration only - in production, models would be saved properly
    print("Note: This is a demonstration API. You would need to save your models as pickle files.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
