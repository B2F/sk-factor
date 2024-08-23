from src.engine.plugins import Plugins
import pandas as pd
from src.engine.config import Config
from src.engine.transfomers import Transformers
from sklearn.preprocessing import LabelEncoder

class Preprocessors():

    @staticmethod
    def apply(config: Config, df: pd.DataFrame) -> pd.DataFrame:
        """ Apply global preprocessing on the DataFrame.
        """

        preprocessors = config.get('preprocess', 'preprocessors')

        if type(preprocessors) is dict:

            for module, features in preprocessors.items():
                preprocessor = Plugins.create(
                    'preprocess.preprocessor', module,
                    config,
                    df,
                    features
                )
                df = preprocessor.transform()

        return df

    @staticmethod
    def encodeLabel(config: Config, df: pd.DataFrame) -> tuple:
        """ Returns preprocessed x, y and decoded labels from the given DataFrame.
        """

        label = config.get('preprocess', 'label')

        y = df[label].to_frame(label)
        x = df
        x = x.drop(label, axis=1)

        labels = y
        if config.eq('preprocess', 'label_encode', True):
            le = LabelEncoder()
            y = pd.DataFrame(list(le.fit_transform(y.values.flatten())), columns=[label])
            labels = le.classes_

        return x, y, labels
