from flask import Flask, request, jsonify, render_template
from feature_extractor import FeatureExtractor
from model_predictor import ModelPredictor
import numpy as np

app = Flask(__name__)

# Load the model
model_path = 'xgboost_heart_classification_model.joblib'
model_predictor = ModelPredictor(model_path)
feature_extractor = FeatureExtractor()

# Symptoms dictionary mapping conditions to their symptoms
symptoms_dict = {
    'Aortic Stenosis (AS)': "Shortness of breath, chest pain, fatigue.",
    'Mitral Stenosis (MS)': "Shortness of breath, fatigue, palpitations.",
    'Mitral Regurgitation (MR)': "Shortness of breath, fatigue, palpitations.",
    'Mitral Valve Prolapse (MVP)': "Chest pain, palpitations, fatigue.",
    'Normal': "No symptoms.",
}

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    """Render the prediction page."""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Check if a file is present in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        # Check if the file is not empty
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected. Please upload a WAV file.'}), 400

        # Save the uploaded file temporarily
        file_path = 'temp.wav'
        file.save(file_path)

        # Extract features from the audio file
        features = feature_extractor.extract_features(file_path)
        if features is None or len(features) == 0:
            return jsonify({'error': 'Feature extraction failed. Please check the audio file.'}), 400

        # Reshape features for prediction
        features = features.reshape(1, -1)

        # Make a prediction
        predicted_class = model_predictor.predict(features)
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed. Please try again.'}), 500

        predicted_class = int(predicted_class[0])

        # Map the predicted class to a condition label
        class_labels = {
            0: 'Aortic Stenosis (AS)',
            1: 'Mitral Stenosis (MS)',
            2: 'Mitral Regurgitation (MR)',
            3: 'Mitral Valve Prolapse (MVP)',
            4: 'Normal'
        }
        predicted_label = class_labels.get(predicted_class, 'Unknown Condition')

        # Get symptoms associated with the predicted condition
        symptoms = symptoms_dict.get(predicted_label, "No symptoms listed.")

        # Return the prediction and symptoms
        return jsonify({
            'prediction': predicted_label,
            'symptoms': symptoms
        })
    except Exception as e:
        # Return a generic error message
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
