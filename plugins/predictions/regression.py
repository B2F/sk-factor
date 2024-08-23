""" Resolve a classifiying problem with predictions filtered by:

    Standard deviation.
"""

import numpy as np
from plugins.predictions.base_predictor import BasePredictor

class Regression(BasePredictor):

    def _predict(self, model):

        # all_predictions = np.array([tree.predict(self._x) for tree in model.estimators_])
        df_predict = model.predict(self._x)
        print(df_predict)
        return df_predict

        # Calculate the mean and standard deviation of the predictions
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        # Keep predictions with a standard deviation below the threshold
        filtered_predictions = mean_predictions[std_predictions < self._threshold]

        # @todo: format table as for multiclass

        return filtered_predictions
