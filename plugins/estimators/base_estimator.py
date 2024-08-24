from abc import ABC, abstractmethod
import pandas as pd
from src.engine.config import Config

class BaseEstimator(ABC):
    """
    Sampler:
    https://imbalanced-learn.org/stable/user_guide.html
    For sampling, use a imblearn pipeline (pipeline = 'imblearn.pipeline' is the training config default)

    Transformer:
    Any transform done before prediction, such as non linear, power transforms
    https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation

    Classifier:
    Sklearn compatible classifier, such as LinearSVC.

    Regressor:
    Sklearn compatible regressor, such as Ridge.

    @see "Choosing the right estimator":
    https://scikit-learn.org/stable/machine_learning_map.html
    """

    TYPES = ('classifier', 'regressor', 'transformer', 'sampler')

    _type = str

    def __init__(self, config: Config, x: pd.DataFrame, y: pd.DataFrame):

        self._config = config
        self._x = x
        self._y = y

    @abstractmethod
    def getEstimator(self) -> tuple:
        '''
        Returns a pipeline tuple such as ('classifier', RandomForestClassifier())
        '''
        pass
