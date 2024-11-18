import joblib

class ModelPredictor:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception:
            self.model = None

    def predict(self, features):
        if self.model is None:
            return None
        return self.model.predict(features)
