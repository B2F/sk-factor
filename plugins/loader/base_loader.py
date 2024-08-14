import pandas as pd
from abc import ABC, abstractmethod

class BaseLoader(ABC):

    _config: dict
    _arguments: list

    def __init__(self, config: dict, arguments: list = []):

        self._arguments = arguments
        self._config = config

    @abstractmethod
    def _load(self) -> pd.DataFrame:

        pass

    def load(self) -> pd.DataFrame:

        df = self._load()
        if self._config.get('dataset', 'show_columns'):

            print(df.columns)

        return df
