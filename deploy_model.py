from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved RobustScaler and the best model
scaler = joblib.load("./model/robust_scaler.pkl")
model = joblib.load("./model/h2o_potable_rnforest.pkl")

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Welcome to the Water Potability Prediction API!",
        "instructions": "To check water potability, send a POST request to /checkpotable with the following parameters:",
        "parameters": [
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
        ]
    })

@app.route("/checkpotable", methods=["POST"])
def check_potable():
    try:
        # Extract data from request
        data = request.get_json()
        features = [
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
        ]
        
        # Validate all required features are provided
        if not all(feature in data for feature in features):
            return jsonify({"error": "Missing one or more required parameters.", "required_features": features}), 400
        
        # Extract feature values
        input_data = np.array([[data[feature] for feature in features]])
        
        # Scale the input data using the RobustScaler
        scaled_data = scaler.transform(input_data)
        
        # Perform inference using the loaded model
        prediction = model.predict(scaled_data)
        
        # Convert prediction to human-readable output
        result = "Potable" if prediction[0] == 1 else "Not Potable"
        
        return jsonify({
            "input_data": data,
            "prediction": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
###
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

