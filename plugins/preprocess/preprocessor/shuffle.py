from plugins.preprocess.base_preprocessor import BasePreprocessor
from pathlib import Path

class Shuffle(BasePreprocessor):

    _id = Path(__file__).stem

    def preprocess(self):

        if type(self._arguments) is not int:
            raise Exception('Shuffle preprocessor plugin takes an int as argument (as random state)')

        return self._df.sample(frac=1, random_state=0).reset_index(drop=True)
