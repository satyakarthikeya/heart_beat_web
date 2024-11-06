from flask import Flask, request, jsonify, render_template
from feature_extractor import FeatureExtractor
from model_predictor import ModelPredictor
import logging
import numpy as np

app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the model using the ModelPredictor class
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests."""
    if request.method == 'POST':
        try:
            # Retrieve the uploaded file
            file = request.files.get('file')
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
            predicted_class = int(predicted_class[0])  # Extract the first element from the prediction

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

            # Prepare an interactive question if the condition is not normal
            if predicted_label != 'Normal':
                questions = f"Do you experience any of the following symptoms: {symptoms}? (yes/no)"
                return jsonify({
                    'prediction': predicted_label,
                    'symptoms': symptoms,
                    'questions': questions
                })

            # Return the prediction and symptoms
            return jsonify({
                'prediction': predicted_label,
                'symptoms': symptoms
            })

        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
