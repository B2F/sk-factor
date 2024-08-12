import pandas as pd

class Data():

    @staticmethod
    def readFile(trainfile: str):

        # @todo allow dynamic excel / csv retrieval.
        df = pd.read_csv(f"data/{trainfile}.csv")
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
