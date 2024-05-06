from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model = load('model.joblib')  # Load the model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000)
