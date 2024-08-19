""" Resolve a classifiying problem with predictions filtered by:

    Confidence threshold
"""

from plugins.predictions.base_predictor import BasePredictor
import pandas as pd

class Multiclass(BasePredictor):

    def _predict(self, model):

        probas = model.predict_proba(self._x)

        estimations = []

        labelRange = range(len(self._labels))
        # Output valid classes given the configured predictions threshold:
        for classes in probas:
            # @todo: per class threshold ?
            validClasses = classes > self._config.get('predictions', 'threshold')
            validLabels = [self._labels[i] for i in labelRange if validClasses[i]]
            validLabels = " or ".join(validLabels)
            if not validLabels:
                estimations.append('Not predicted')
            else:
                estimations.append(validLabels)

        testRange = range(len(estimations))
        probasDf = pd.DataFrame(data = probas, columns = self._labels)
        estimations = pd.DataFrame(data = estimations, columns=['estimation'])
        threshold = pd.DataFrame(data = [self._threshold for _ in testRange], columns=['threshold'])

        outputColumns = [probasDf, threshold, estimations]
        if self._config.eq('predictions', 'keep_data', True):
            outputColumns = [self._x] + outputColumns

        predictions_output = pd.concat(outputColumns, sort=False, axis="columns")

        print(predictions_output)

        return predictions_output
