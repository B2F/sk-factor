from plugins.preprocess.base_preprocessor import BasePreprocessor
from pathlib import Path
import pandas as pd

class DropColumns(BasePreprocessor):

    _id = Path(__file__).stem

    def preprocess(self):

        dfColumns = list(self._df.columns)
        if type(self._arguments) is list:
            for feature in self._arguments:
                dfColumns.remove(feature)
        else:
            dfColumns.remove(self._arguments)

        self._df = self._df[dfColumns]

        return self._df
