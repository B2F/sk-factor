from src.engine.plugins import Plugins
import pandas as pd

class Preprocessors():

    @staticmethod
    def apply(preprocessors: dict, df: pd.DataFrame):

        print('Before preprocessing:')
        print(df.shape)

        if preprocessors is not dict:

            return df

        for action, features in preprocessors.items():
            preprocessor = Plugins.create('preprocess', 'preprocessor/' + action, df, features)
            df = preprocessor.transform()

        return df
