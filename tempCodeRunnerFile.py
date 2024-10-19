@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        
        # Save the uploaded file temporarily
        file_path = 'temp.wav'
        file.save(file_path)
        
        # Extract features from the audio file using the feature extractor
        features = feature_extractor.extract_features(file_path)
        
        # Check if features are empty
        if features is None or len(features) == 0:
            return jsonify({'error': 'Feature extraction failed. Please check the audio file.'}), 400
        
        # Reshape features if needed
        features = features.reshape(1, -1)
        
        # Make a prediction using the model predictor class
        predicted_class = model_predictor.predict(features)
        
        # Convert the prediction result to a standard Python int or str
        predicted_class = int(predicted_class[0])  # Assuming it's an array, extract the first element
        
        # Get the corresponding class label (e.g., 'Aortic Stenosis (AS)', 'Normal', etc.)
        class_labels = {
            0: 'Aortic Stenosis (AS)',
            1: 'Mitral Stenosis (MS)',
            2: 'Mitral Regurgitation (MR)',
            3: 'Mitral Valve Prolapse (MVP)',
            4: 'Normal'
        }
        predicted_label = class_labels.get(predicted_class, 'Unknown Condition')
        
        # Get symptoms associated with the prediction
        symptoms = symptoms_dict.get(predicted_label, "No symptoms listed.")
        
        # Ask about symptoms interactively
        if predicted_label != 'Normal':
            questions = f"Do you experience any of the following symptoms: {symptoms}? (yes/no)"
            return jsonify({
                'prediction': predicted_label,
                'symptoms': symptoms,
                'questions': questions
            })
        
        return jsonify({
            'prediction': predicted_label,
            'symptoms': symptoms
        })
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
