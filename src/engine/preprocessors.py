from src.engine.plugins import Plugins
import pandas as pd
from src.engine.config import Config
from src.engine.transfomers import Transformers

class Preprocessors():

    @staticmethod
    def apply(config: Config, df: pd.DataFrame) -> tuple:
        """ Returns preprocessed x, y and decoded labels from the given DataFrame.
        """

        print('Before preprocessing:')
        print(df.shape)

        preprocessors = config.get('preprocess', 'preprocessors')
        label = config.get('preprocess', 'label')

        if type(preprocessors) is dict:

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
        x = x.drop(label, axis=1)

        labels = y
        if config.eq('preprocess', 'label_encode', True):
            y, labels = Transformers.labelEncode(y, label)

        return x, y, labels
