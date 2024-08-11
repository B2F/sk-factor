from preprocess.base_preprocessor import BasePreprocessor
from pathlib import Path

class DropColumn(BasePreprocessor):

    _id = Path(__file__).stem

    def preprocess(self, columns: list):

        dfColumns = list(columns)
        for feature in columns:
            dfColumns.remove(feature)

        df = df[dfColumns]

        return self._df
