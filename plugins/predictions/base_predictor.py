from abc import ABC, abstractmethod
import pandas as pd
from src.engine.model import Model
from src.engine.files import Files

class BasePredictor(ABC):

    _config: dict
    _x: pd.DataFrame
    _threshold: int

    def __init__(self, config, x):

        self._config = config
        self._x = x
        self._threshold = config.get('predictions', 'threshold')

    @abstractmethod
    def _predict(self, model: Model):
        """Calculate predictions here.
        You can add your own predictions objective plugin.

        Default strategies are different depending on the target type:

                .) Classifier -> confidence_threshold
                .) Regression -> standard_deviation

        Returns array-like data provided by methods such as predict_proba
        """
        pass

    def predict(self, model: Model):

        predictions = self._predict(model)
        if self._config.eq('predictions', 'save_predictions', True):
            directory = self._config.get('predictions', 'predictions_directory')
            filename = model.id
            withTimestamp = self._config.get('predictions', 'predictions_timestamp')
            Files.toCsv(predictions, directory, filename, withTimestamp)
