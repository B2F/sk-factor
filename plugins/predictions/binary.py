""" Resolve a binary classification problem with predictions filtered by:

    Confidence threshold
"""

from plugins.predictions.base_predictor import BasePredictor
from src.engine.model import Model
import pandas as pd

class Binary(BasePredictor):

    def _predict(self, model: Model):

        probas = model.pipeline.predict_proba(self._x)[:, 1]

        # Output valid classes given the configured predictions threshold:
        estimations = probas > self._config.get('predictions', 'threshold')

        testRange = range(len(estimations))
        probasDf = pd.DataFrame(data = probas, columns = ['proba'])
        threshold = pd.DataFrame(data = [self._threshold for _ in testRange], columns=['threshold'])
        estimations = pd.DataFrame(data = estimations, columns=['estimation'])

        outputColumns = [probasDf, threshold, estimations]
        if self._config.eq('predictions', 'keep_data', True):
            outputColumns = [self._x] + outputColumns

        predictions_output = pd.concat(outputColumns, sort=False, axis="columns")

        print(predictions_output)

        return predictions_output
