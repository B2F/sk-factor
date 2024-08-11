import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

class BasePreprocessor(ABC):

    _df = pd.DataFrame
    _config = dict
    _id = Path(__file__).stem

    def __init__(self, config, df: pd.DataFrame):
        self._config = config
        self._df = df

    @abstractmethod
    def preprocess(self, features: list) -> pd.DataFrame:
        # Preprocess the whole dataset at once here.
        return self._df

    def transform(self):
        print('\n' + self._id + ':')
        df = self.preprocess()
        print(df.shape)
        return df
