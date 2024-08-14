from plugins.preprocess.base_preprocessor import BasePreprocessor
from pathlib import Path
import pandas as pd

class DropColumns(BasePreprocessor):

    _id = Path(__file__).stem
    _columns: list

    def __init__(self, config, df: pd.DataFrame, features: list = None):
        self._config = config
        self._df = df
        self._columns = features

    def preprocess(self):

        dfColumns = list(self._df.columns)
        for feature in self._columns:
            dfColumns.remove(feature)

        self._df = self._df[dfColumns]

        return self._df
