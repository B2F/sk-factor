import pandas as pd
import os

class Data():

    _DEFAULT_DIRECTORY = 'data/'

    @staticmethod
    def readFile(trainfile: str):

        if trainfile.find('.csv', -4) == -1:
            trainfile = Data._DEFAULT_DIRECTORY + trainfile + '.csv'

        if not os.path.isfile(trainfile):
            raise Exception(trainfile + ' not found.')

        # @todo allow dynamic excel / csv retrieval.
        df = pd.read_csv(f"{trainfile}")
        return df

    @staticmethod
    def addGroup(df: pd.DataFrame, groupName: str):

        df['group'] = groupName
        return df

    @staticmethod
    def mergeFiles(dataFrames: list, axis: str):

        if axis not in ['index', 'column']:
            raise Exception('trainfiles_axis can only be either index or column')
        mapAxis = 0 if axis == 'index' else 1

        return pd.concat(dataFrames, axis = mapAxis)
