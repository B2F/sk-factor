""" Resolve a classifiying problem with predictions filtered by:

    Confidence threshold
"""

from plugins.predictions.base_predictor import BasePredictor

class Multiclass(BasePredictor):

    def predict(self, model):

        probas = model.predict_proba(self._x)
        return probas
        # Adjust predictions based on the threshold
        adjusted_predictions = (probas[:, 1] >= self._threshold).astype(int)

        return adjusted_predictions
