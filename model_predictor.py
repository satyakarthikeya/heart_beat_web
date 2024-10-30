import joblib
import numpy as np
import logging

class ModelPredictor:
    def __init__(self, model_path):
        # Load the model
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Loading model from: {model_path}")
        try:
            self.model = joblib.load(model_path)
            self.logger.debug("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            self.model = None

    def predict(self, features):
        try:
            if self.model is None:
                raise ValueError("Model is not loaded.")
            return self.model.predict(features)
        except Exception as e:
            logging.error('Error during model prediction: %s', str(e))
            return None
