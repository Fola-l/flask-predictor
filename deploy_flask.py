import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# -------------------------------
# Load Models
# -------------------------------
MODEL_PATH = "models"

model_files = {
    "DecisionTree": "DecisionTree_model.pkl",
    "GradientBoosting": "GradientBoosting_model.pkl",
    "LogisticRegression": "LogisticRegression_model.pkl",
    "RandomForest": "RandomForest_model.pkl",
}

models = {}

for name, file in model_files.items():
    path = os.path.join(MODEL_PATH, file)
    try:
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"Loaded {name}")
    except Exception as e:
        print(f"Error loading {name}: {e}")


# -------------------------------
# Health Check Route
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask ML API is running 🚀"})


# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    input_data = request.get_json()

    sample = input_data.get("data")
    if sample is None:
        return jsonify({"error": "Key 'data' not found in JSON"}), 400

    try:
        # Convert input to DataFrame
        df = pd.DataFrame(sample if isinstance(sample, list) else [sample])
    except Exception as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

    predictions = {}

    for name, model in models.items():
        try:
            preds = model.predict(df)
            predictions[name] = preds.tolist()
        except Exception as e:
            predictions[name] = f"Prediction error: {e}"

    return jsonify({"predictions": predictions})


# -------------------------------
# Run locally only
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
