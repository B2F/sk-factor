import os
import pandas as pd
from plugins.loader.base_loader import BaseLoader

class Csv(BaseLoader):

    _DEFAULT_DIRECTORY = 'data/'

    def load(self):

        mergeAxis = self._config.get('preprocess', 'files_axis')
        dataFrames = list(map(self.readCsv, self._arguments))
        if self._config.eq(*('preprocess', 'groupFiles'), True):
            dataFrames = list(map(self.addGroup, *(dataFrames, 'group')))
        df = self.mergeFiles(dataFrames, mergeAxis)
        return df

    def readCsv(self, trainfile):

        if trainfile.find('.csv', -4) == -1:
            trainfile = self._DEFAULT_DIRECTORY + trainfile + '.csv'

        if not os.path.isfile(trainfile):
            raise Exception(trainfile + ' not found.')

        df = pd.read_csv(f"{trainfile}")
        return df

    def addGroup(self, df: pd.DataFrame, groupName: str):

        df['group'] = groupName
        return df

    def mergeFiles(self, dataFrames: list, axis: str):

        if axis not in ['index', 'column']:
            raise Exception('trainfiles_axis can only be either index or column')
        mapAxis = 0 if axis == 'index' else 1

        return pd.concat(dataFrames, axis = mapAxis)
