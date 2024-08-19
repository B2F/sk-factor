""" Resolve a binary classification problem with predictions filtered by:

    Confidence threshold
"""

from plugins.predictions.base_predictor import BasePredictor

class Binary(BasePredictor):

    def _predict(self, model):

        probas = model.predict_proba(self._x)

        # Adjust predictions based on the threshold
        adjusted_predictions = (probas[:, 1] >= self._threshold).astype(int)

        return adjusted_predictions
