from plugins.preprocess.base_preprocessor import BasePreprocessor
from pathlib import Path

class DropRows(BasePreprocessor):

    _id = Path(__file__).stem

    def preprocess(self):

        if type(self._arguments) is not int:
            raise Exception('Drop rows preprocessor plugin takes an int as argument (as remove count)')

        count = self._arguments
        reducedDataset = self._df.iloc[:count]

        dropToPredictFile = self._config.get('preprocess', 'drop_rows_to_predict_file')
        suffix = 'for predictions' if dropToPredictFile else ''
        if count > 0:
            removedDf = self._df.head(count)
            print (f'...\nFirst {count} rows removed {suffix}:')
        else:
            removedDf = self._df.tail(count*-1)
            print (f'...\nLast {count*-1} rows removed {suffix}:')

        print(removedDf)

        return reducedDataset
