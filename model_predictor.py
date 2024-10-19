import joblib
import numpy as np
import logging
class ModelPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        try:
            return self.model.predict(features)
        except Exception as e:
            logging.error('Error during model prediction: %s', str(e))
            return None
