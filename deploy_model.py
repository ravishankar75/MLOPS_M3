
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('linear_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    predictions = model.predict([data['features']])
    return jsonify(predictions=predictions.tolist())

if __name__ == "__main__":
    app.run(debug=True)
