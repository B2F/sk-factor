from src.engine.plugins import Plugins
import pandas as pd
from src.engine.config import Config

class Preprocessors():

    @staticmethod
    def apply(preprocessors: dict, config: Config, df: pd.DataFrame, label: str) -> tuple:
        """ Returns x and y from a given DataFrame.
        """

        print('Before preprocessing:')
        print(df.shape)

        if type(preprocessors) is not dict:

            return df

        for module, features in preprocessors.items():
            preprocessor = Plugins.create(
                'preprocess.preprocessor', module,
                config,
                df,
                features
            )
            df = preprocessor.transform()

        y = df[label].to_frame(label)
        x = df
        x.drop(label, axis=1, inplace=True)

        return x, y
