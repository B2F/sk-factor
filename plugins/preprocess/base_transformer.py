from abc import ABC
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator

class BaseTransformer(ABC):
    """ Feature preprocessing plugins.
    https://scikit-learn.org/stable/modules/preprocessing.html
    """

    _name: str
    _config = dict
    _df = pd.DataFrame

    def __init__(self, config: dict, df: pd.DataFrame):

        self._config = config
        self._df = df

        if self._name is None:
            raise Exception('Transformer plugin needs a _name property.')

    def estimator(self) -> BaseEstimator:
        return FunctionTransformer()

    def getSteps(self) -> list:
        return [
            (self._name, self.estimator())
        ]

    def pipeline(self) -> Pipeline:
        return Pipeline(
            steps=self.getSteps()
        )
