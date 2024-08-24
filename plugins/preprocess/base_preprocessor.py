import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

class BasePreprocessor(ABC):
    """ DataFrame preprocessor.
    """

    _df = pd.DataFrame
    _config = dict
    _id = Path(__file__).stem
    _arguments: list

    def __init__(self, config, df: pd.DataFrame, arguments = None):
        self._config = config
        self._df = df
        self._arguments = arguments

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        # Preprocess the whole dataset at once here.
        return self._df

    def transform(self):
        print('\nAfter ' + self._id + ':')
        df = self.preprocess()
        print(df.shape)
        return df
