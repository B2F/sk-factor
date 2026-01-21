from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

class BaseYPreprocessor(ABC):
    """ Y label preprocessor - handles transformations on the target variable.
    """

    _y = pd.DataFrame
    _config = dict
    _id = Path(__file__).stem
    _arguments = None

    def __init__(self, config, y: pd.DataFrame, arguments=None):
        self._config = config
        self._y = y
        self._arguments = arguments

    @abstractmethod
    def preprocess_y(self) -> pd.DataFrame:
        """ Preprocess the Y (target) variable.

        Returns:
            pd.DataFrame: The transformed Y variable
        """
        return self._y

    def transform(self):
        """ Apply the transformation and print debug info.

        Returns:
            pd.DataFrame: The transformed Y variable
        """
        print(f'\nAfter {self._id} on Y:')
        y_transformed = self.preprocess_y()
        print(f"Y shape: {y_transformed.shape}")
        if len(y_transformed.columns) > 0:
            print(f"Y columns: {list(y_transformed.columns)}")
            print(f"Sample values:\n{y_transformed.head()}")
        return y_transformed
