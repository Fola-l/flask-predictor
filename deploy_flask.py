from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

models = {}
model_files = {
    "DecisionTree": os.path.join("models", "DecisionTree_model.pkl"),
    "GradientBoosting": os.path.join("models", "GradientBoosting_model.pkl"),
    "LogisticRegression": os.path.join("models", "LogisticRegression_model.pkl"),
    "RandomForest": os.path.join("models", "RandomForest_model.pkl"),
}

load_errors = {}

for name, filepath in model_files.items():
    try:
        models[name] = joblib.load(filepath)
        print(f"Loaded {name} from {filepath}")
    except Exception as e:
        load_errors[name] = f"{type(e).__name__}: {e}"
        print(f"Error loading {name}: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "loaded_models": list(models.keys()),
        "load_errors": load_errors
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not models:
        return jsonify({
            "error": "No models loaded on server",
            "load_errors": load_errors
        }), 500

    input_data = request.get_json(silent=True)
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    sample = input_data.get("data")
    if sample is None:
        return jsonify({"error": "Key 'data' not found in JSON"}), 400

    try:
        df = pd.DataFrame(sample if isinstance(sample, list) else [sample])
    except Exception as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

    predictions = {}
    for name, model in models.items():
        try:
            preds = model.predict(df)
            predictions[name] = preds.tolist()
        except Exception as e:
            predictions[name] = f"{type(e).__name__}: {e}"

    return jsonify({"predictions": predictions})
