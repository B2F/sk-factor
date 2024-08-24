""" Resolve a classifiying problem with predictions filtered by:

    Standard deviation.
"""

import pandas as pd
from plugins.predictions.base_predictor import BasePredictor
from src.engine.model import Model

class Regression(BasePredictor):

    def _predict(self, model: Model):

        df_predict = model.pipeline.predict(self._x)

        estimations = pd.DataFrame(data = df_predict, columns=['prediction'])

        outputColumns = [estimations]
        if self._config.eq('predictions', 'keep_data', True):
            outputColumns = [self._x] + outputColumns

        predictions_output = pd.concat(outputColumns, sort=False, axis="columns")

        print(predictions_output)

        return predictions_output
