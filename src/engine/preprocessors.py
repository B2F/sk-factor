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

        # Apply Y preprocessors if configured
        y_preprocessors = config.get('preprocess', 'y_preprocessors')
        if type(y_preprocessors) is list:
            for module in y_preprocessors:
                y_preprocessor = Plugins.create(
                    'preprocess.y_preprocessor', module,
                    config,
                    y
                )
                y = y_preprocessor.transform()

        # Maintain backward compatibility with label_encode flag
        labels = y
        if config.eq('preprocess', 'label_encode', True):
            # If label_encode is True and no y_preprocessors are configured,
            # use the default label encoder for backward compatibility
            if y_preprocessors is None or len(y_preprocessors) == 0:
                le = LabelEncoder()
                y = pd.DataFrame(list(le.fit_transform(y.values.flatten())), columns=[label])
                labels = le.classes_
        else:
            # If label_encode is not True, store original labels
            if hasattr(y, 'values'):
                labels = pd.Series(y.values.flatten()).unique()

        return x, y, labels
