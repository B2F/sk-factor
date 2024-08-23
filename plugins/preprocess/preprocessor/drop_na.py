from plugins.preprocess.base_preprocessor import BasePreprocessor
from pathlib import Path

class DropNa(BasePreprocessor):

    _id = Path(__file__).stem

    def preprocess(self):

        if self._arguments is not None:
            self._df.dropna(axis=0, inplace=True, subset=self._arguments)
        else:
            self._df.dropna(axis=0, inplace=True)

        self._df.reset_index(inplace=True)
        return self._df
