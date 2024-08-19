from abc import abstractmethod
import pandas as pd

class BasePredictor():

    _x: pd.DataFrame
    _threshold: int

    def __init__(self, config, x):

        self._x = x
        self._threshold = config.get('predictions', 'threshold')

    @abstractmethod
    def predict(self, model: object):
        """Calculate predictions here.
        You can add your own predictions objective plugin.

        Default strategies are different depending on the target type:

                .) Classifier -> confidence_threshold
                .) Regression -> standard_deviation

        Returns array-like data provided by methods such as predict_proba
        """
        pass
