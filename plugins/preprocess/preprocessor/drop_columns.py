from plugins.preprocess.base_preprocessor import BasePreprocessor
from pathlib import Path

class DropColumns(BasePreprocessor):

    _id = Path(__file__).stem

    def preprocess(self):

        dfColumns = list(self._df.columns)
        if type(self._arguments) is list:
            for feature in self._arguments:
                # _arguments can be missing, on Y for instance:
                if feature in dfColumns:
                    dfColumns.remove(feature)
        elif self._arguments in dfColumns:
            dfColumns.remove(self._arguments)

        self._df = self._df[dfColumns]

        return self._df
