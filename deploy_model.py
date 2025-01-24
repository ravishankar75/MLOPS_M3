from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model, scaler, and imputer
model = joblib.load('water_potability_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        input_data = request.json

        if not input_data or 'features' not in input_data:
            return jsonify({'error': 'Missing "features" in request data'}), 400

        # Extract features from input JSON
        features = np.array(input_data['features']).reshape(1, -1)

        # Apply imputation and scaling
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)

        # Make prediction
        prediction = model.predict(features_scaled)

        result = "potable" if prediction[0] == 1 else "not potable"
        
        return jsonify({'prediction': result, 'input_features': input_data['features']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

