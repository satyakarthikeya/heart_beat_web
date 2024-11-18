from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)

# Load the XGBoost model
model_path = "xgboost_heart_classification_model.json"  # Use the .json model file
model = xgb.Booster()
model.load_model(model_path)

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        # Retrieve the uploaded file
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No file selected. Please upload a WAV file."}), 400

        # Save the uploaded file temporarily
        file_path = "temp.wav"
        file.save(file_path)

        # Extract features from the audio file (you need to implement this)
        features = extract_features(file_path)  # Make sure extract_features is defined
        if features is None or len(features) == 0:
            return jsonify({"error": "Feature extraction failed. Please check the audio file."}), 400

        # Reshape features for prediction
        features = features.reshape(1, -1)

        # Make a prediction
        dmatrix = xgb.DMatrix(features)
        predicted_class = model.predict(dmatrix)
        predicted_class = int(np.round(predicted_class[0]))

        # Map the predicted class to a condition label
        class_labels = {
            0: "Aortic Stenosis (AS)",
            1: "Mitral Stenosis (MS)",
            2: "Mitral Regurgitation (MR)",
            3: "Mitral Valve Prolapse (MVP)",
            4: "Normal"
        }
        predicted_label = class_labels.get(predicted_class, "Unknown Condition")

        # Symptoms dictionary mapping conditions to their symptoms
        symptoms_dict = {
            "Aortic Stenosis (AS)": "Shortness of breath, chest pain, fatigue.",
            "Mitral Stenosis (MS)": "Shortness of breath, fatigue, palpitations.",
            "Mitral Regurgitation (MR)": "Shortness of breath, fatigue, palpitations.",
            "Mitral Valve Prolapse (MVP)": "Chest pain, palpitations, fatigue.",
            "Normal": "No symptoms."
        }
        symptoms = symptoms_dict.get(predicted_label, "No symptoms listed.")

        # Return the prediction and symptoms
        return jsonify({
            "prediction": predicted_label,
            "symptoms": symptoms
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
