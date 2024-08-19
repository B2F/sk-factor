from abc import ABC, abstractmethod
import os
import time
import pandas as pd

class BasePredictor(ABC):

    _config: dict
    _x: pd.DataFrame
    _threshold: int
    _labels: list

    def __init__(self, config, x, labels):

        self._config = config
        self._x = x
        self._threshold = config.get('predictions', 'threshold')
        self._labels = labels

    @abstractmethod
    def _predict(self, model: object):
        """Calculate predictions here.
        You can add your own predictions objective plugin.

        Default strategies are different depending on the target type:

                .) Classifier -> confidence_threshold
                .) Regression -> standard_deviation

        Returns array-like data provided by methods such as predict_proba
        """
        pass

    def predict(self, model: object):

        predictions = self._predict(model)
        if self._config.eq('predictions', 'save_predictions', True):
            self._writeToFile(predictions)

    def _writeToFile(self, predictions):

        directory = self._config.get('predictions', 'predictions_directory')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        prediction_filename = self._config.get('dataset', 'filename')

        if self._config.eq('predictions', 'predictions_timestamp', True):
            prediction_filename += '_' + str(time.time())

        fullPath = f"{directory}/{prediction_filename}.csv"
        predictions.to_csv(f"{fullPath}")

        print (f'{fullPath} written to disk.')